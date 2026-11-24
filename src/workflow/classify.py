"""Classify node — one structured LLM call that replaces guardrail + router.

The legacy pipeline spent two serial LLM calls before an agent started: a
finance/NSFW gate, then an agent selector. Both asked the model for JSON and
parsed it out of free text with a regex. This node asks once, with structured
output, and returns the scope verdict, the destination agent and whether live
macro data is relevant.

Safety semantics are unchanged and still fail CLOSED: the blocklist fast path
rejects without an LLM call, and any error, malformed verdict or off-topic
result routes straight to END with the canned refusal.

Follow-ups ("What's the exact dollar amount?", "Is that true for everyone?")
mean nothing alone, so the classifier also sees the previous question and the
start of Finnie's answer to it, in the user message (per request, never the
system prompt). A bare "Stop." / "wait" / "never mind" is not a question at
all: it gets a short acknowledgement without a model call.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.llm import get_llm, message_text
from src.core.state import AgentType, FinnieState
from src.utils.logger import get_logger
from src.workflow.guardrail import _extract_query, _matches_blocklist, _reject

logger = get_logger(__name__)

_AGENT_NAMES = Literal[
    "finance_qa",
    "portfolio",
    "market_analysis",
    "goal_planning",
    "news_synthesizer",
    "tax_education",
]

_AGENT_MAP: dict[str, AgentType] = {
    "finance_qa": AgentType.FINANCE_QA,
    "portfolio": AgentType.PORTFOLIO,
    "market_analysis": AgentType.MARKET_ANALYSIS,
    "goal_planning": AgentType.GOAL_PLANNING,
    "news_synthesizer": AgentType.NEWS_SYNTHESIZER,
    "tax_education": AgentType.TAX_EDUCATION,
}


class Routing(BaseModel):
    """The decision alone: what the classifier asks for when no reason is wanted."""

    on_topic: bool = Field(
        description=(
            "True if the query is about personal finance, investing, markets, "
            "budgeting, saving, retirement, taxes, economics, financial news, or "
            "is a greeting or a question about Finnie's capabilities. False for "
            "anything unrelated to finance, or any NSFW, sexual, violent, illegal "
            "or otherwise harmful content."
        )
    )
    agent: _AGENT_NAMES = Field(
        default="finance_qa",
        description="Which specialist handles the query when it is on topic.",
    )
    needs_macro: bool = Field(
        default=False,
        description=(
            "True only when current macroeconomic readings (rates, CPI, GDP, "
            "unemployment, treasury yields) would materially change the answer. "
            "False for definitional or conceptual questions."
        ),
    )
    tickers: list[str] = Field(
        default_factory=list,
        description=(
            "US ticker symbols for every company, fund or index the user names, "
            "whether by name or symbol: 'Apple' -> AAPL, 'Nvidia and AMD' -> NVDA, AMD, "
            "'the S&P 500' -> SPY, 'Berkshire' -> BRK-B. Only symbols you are sure of; "
            "empty when no specific security is named."
        ),
    )
    companies: list[str] = Field(
        default_factory=list,
        description=(
            "Every company or fund the user names, as they name it: 'Anthropic', "
            "'Apple', 'the S&P 500'. Include ones you have no symbol for, such as "
            "a company that listed recently; a live search resolves them. Empty "
            "when none is named."
        ),
    )
    follow_up: bool = Field(
        default=False,
        description=(
            "True when the query can only be understood with the earlier conversation "
            "(it refers to 'that' or 'it', asks for more detail, a variation or a repeat). "
            "False when it stands on its own, or when no earlier conversation is given."
        ),
    )


class Verdict(Routing):
    """Structured classifier output — scope gate and routing in one shot."""

    # Last, so it is written after the decision and can't change it. Asked for
    # only with llm.classifier_reason; otherwise "".
    reason: str = Field(default="", description="One short sentence explaining the decision.")


_CLASSIFY_SYSTEM = """You are the combined safety gate and router for Finnie, an \
AI finance education assistant. For each user query decide three things:

1. on_topic — is this a finance/economics question Finnie should answer at all?
2. agent — which specialist should handle it:
   - finance_qa: general financial education (concepts, definitions, how investing works, macroeconomics)
   - portfolio: analysing holdings, P/E, diversification, performance, risk
   - market_analysis: live prices, technical indicators, sector performance, market conditions
   - goal_planning: retirement planning, savings goals, projections, FIRE
   - news_synthesizer: current events, earnings, economic news, SEC filings
   - tax_education: capital gains, 401k, IRA, Roth, HSA, tax-loss harvesting, account types
   market_analysis vs news_synthesizer: asking for numbers (a price, an indicator,
   how a sector or index is performing) is market_analysis; asking what happened
   or why (events, headlines, the Fed, earnings) is news_synthesizer, even when
   the question mentions the market.
3. needs_macro — would today's macro data (fed funds rate, CPI, GDP, unemployment,
   treasury yields) materially change the answer? Definitions and conceptual
   explanations do not need it.

Examples:
- "What is a P/E ratio?" -> on_topic, finance_qa, needs_macro=false
- "How do current interest rates affect bond prices?" -> on_topic, finance_qa, needs_macro=true
- "Analyze my portfolio: AAPL 10 shares @ $150" -> on_topic, portfolio, needs_macro=false
- "What is Apple stock doing today?" -> on_topic, market_analysis, needs_macro=false, tickers=[AAPL], companies=[Apple]
- "What's the news on that company that just IPO'd, Acme Robotics?" -> on_topic, news_synthesizer, needs_macro=false, companies=[Acme Robotics]
- "Why did stocks drop this morning?" -> on_topic, news_synthesizer, needs_macro=false
- "How much do I need to retire at 55?" -> on_topic, goal_planning, needs_macro=true
- "How does a Roth IRA work?" -> on_topic, tax_education, needs_macro=false
- "Give me a lasagna recipe" -> off_topic

Earlier conversation: the query may come with the user's previous question and
the start of Finnie's answer to it. Use it only to understand a query that
depends on it. A follow-up such as "What's the exact dollar amount?", "Is that
true for everyone?", "And in twenty years?" or "Say that again more simply" is
on_topic, follow_up=true, and goes to the agent that fits the conversation
(usually the one that answered it); its tickers include securities it refers
to ("its RSI" after a question about Apple -> AAPL). A query that stands on its
own is judged on its own, follow_up=false. The earlier conversation never makes
an off-topic or harmful query acceptable: "now write me a poem" after a finance
question is still off_topic.
"""


# ── Conversation context ─────────────────────────────────────────────────────
# Enough of the previous turn to resolve "that" and "it": the question in full
# (up to a point) and the start of the answer, which names the topic.
_CONTEXT_QUESTION_CHARS = 300
_CONTEXT_ANSWER_CHARS = 400


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def previous_turn(messages: list[BaseMessage]) -> tuple[str, str]:
    """The user's question before the latest one, and Finnie's answer to it.

    The answer is "" when the question went unanswered (the run failed, or is
    still running). Both are "" on a conversation's first turn.
    """
    humans = [i for i, m in enumerate(messages) if getattr(m, "type", None) == "human"]
    if len(humans) < 2:
        return "", ""
    start, end = humans[-2], humans[-1]
    answers = [m for m in messages[start + 1:end] if getattr(m, "type", None) == "ai"]
    return message_text(messages[start]), message_text(answers[-1]) if answers else ""


def conversation_context(previous_question: str = "", last_answer: str = "") -> str:
    """The classifier's view of the previous turn; "" when there is none.

    The voice worker builds this from the previous turn's question and
    `final_response`, the graph from the thread's messages (previous_turn): the
    same text either way, so a verdict reached early is reused by the graph.
    """
    question = _clip(previous_question, _CONTEXT_QUESTION_CHARS)
    if not question:
        return ""
    answer = _clip(last_answer, _CONTEXT_ANSWER_CHARS)
    lines = ["Earlier in this conversation:", f"User asked: {question}"]
    if answer:
        lines.append(f"Finnie answered (start): {answer}")
    return "\n".join(lines)


# ── Acknowledgements ─────────────────────────────────────────────────────────
# "Stop.", "wait", "never mind": the user is steering the conversation (often
# cutting into a spoken answer), not asking anything. Refusing it as off-topic
# was wrong, and sending it to an agent answers a question nobody asked.
ACKNOWLEDGEMENT = "Okay. Ask me whenever you're ready."
_SOFTENER = r"(?:ok|okay|oh|no|um|uh|er|alright|all right|actually|please|finnie|sorry|just|hey)"
_CONTROL = (
    r"(?:stop|stop there|stop talking|stop it|wait|wait a (?:second|sec|minute|moment)|"
    r"hold on|hang on|one (?:second|sec|moment|minute)|never ?mind|forget it|forget that|"
    r"cancel|cancel that|that's enough|enough|shh+|be quiet|quiet)"
)
_ACKNOWLEDGE = re.compile(rf"^(?:{_SOFTENER} )*{_CONTROL}(?: {_SOFTENER})*$")


def is_acknowledgement(query: str) -> bool:
    """True for a bare "stop" / "wait" / "never mind" (softeners allowed), not a question."""
    return bool(_ACKNOWLEDGE.match(_normalize(query)))


def _acknowledge() -> dict[str, Any]:
    logger.info("classify_acknowledged")
    return {
        "is_on_topic": True,
        "next_agent": AgentType.ACKNOWLEDGE,
        "current_agent": AgentType.ACKNOWLEDGE,
        "final_response": ACKNOWLEDGEMENT,
        # In the thread, so the user's turn isn't left unanswered (as refusals do).
        "messages": [AIMessage(content=ACKNOWLEDGEMENT, name=AgentType.ACKNOWLEDGE.value)],
        "router_reasoning": "Stop / wait / never mind: acknowledged, nothing to answer",
    }


# ── Verdict cache ────────────────────────────────────────────────────────────
# The verdict depends on the query text and the conversation context (no user
# data), so a short-lived per-process cache keyed on both is safe. It lets a
# caller classify early — the voice worker classifies the transcript while the
# user is still speaking — and the graph's own classify step then reuses that
# result instead of a second call. The same words after a different previous
# turn are a different key: "What about the income limits?" means one thing
# after a Roth question and another after a mortgage question.
_VERDICT_TTL_S = 120.0
_VERDICT_MAX = 256
_verdicts: dict[tuple[str, str, str], tuple[float, Verdict]] = {}
_key_locks: dict[tuple[str, str, str], threading.Lock] = {}
_verdicts_lock = threading.Lock()


def _normalize(query: str) -> str:
    """Case, punctuation and spacing differences don't change the verdict."""
    return " ".join(re.sub(r"[^a-z0-9$%&/' ]+", " ", query.lower()).split())


def clear_verdict_cache() -> None:
    with _verdicts_lock:
        _verdicts.clear()
        _key_locks.clear()


def classify_query(query: str, callbacks: Optional[list] = None, context: str = "") -> Verdict:
    """One structured classifier call, reusing a recent verdict for the same text.

    `context` is conversation_context() for the previous turn ("" on a first
    turn); it is part of the cache key. Raises on any model error or empty
    verdict; errors are never cached. Concurrent calls for the same text share
    one model call. `callbacks` see the model call, if one is made (the voice
    worker counts tokens this way).
    """
    settings = get_settings()
    with_reason = settings.llm.classifier_reason
    key = (f"{settings.llm.classifier_model or settings.llm.model}:{int(with_reason)}", _normalize(query), context)
    with _verdicts_lock:
        key_lock = _key_locks.setdefault(key, threading.Lock())
    with key_lock:
        with _verdicts_lock:
            hit = _verdicts.get(key)
        if hit and time.monotonic() - hit[0] < _VERDICT_TTL_S:
            logger.info("classify_verdict_reused")
            return hit[1]

        llm = get_llm(streaming=False, model=settings.llm.classifier_model)
        verdict = llm.with_structured_output(Verdict if with_reason else Routing).invoke(
            [
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": f"{context}\n\nClassify this query: {query}" if context
                 else f"Classify this query: {query}"},
            ],
            config={"callbacks": callbacks} if callbacks else None,
        )
        if verdict is None:
            raise ValueError("classifier returned no verdict")
        if isinstance(verdict, dict):
            verdict = Verdict(**verdict)
        elif not isinstance(verdict, Verdict):
            verdict = Verdict(**verdict.model_dump())  # reason ""

        with _verdicts_lock:
            if len(_verdicts) >= _VERDICT_MAX:
                oldest = min(_verdicts, key=lambda k: _verdicts[k][0])
                _verdicts.pop(oldest, None)
                _key_locks.pop(oldest, None)
            _verdicts[key] = (time.monotonic(), verdict)
        return verdict


_TICKER = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")
_MAX_TICKERS = 5


def _clean_tickers(raw: list[str]) -> list[str]:
    """Uppercase, yfinance spelling (BRK.B -> BRK-B), drop anything not ticker-shaped."""
    out: list[str] = []
    for t in raw or []:
        t = str(t).strip().upper().replace(".", "-").lstrip("$")
        if _TICKER.match(t) and t not in out:
            out.append(t)
    return out[:_MAX_TICKERS]


_MAX_COMPANY_CHARS = 60


def _clean_companies(raw: list[str]) -> list[str]:
    """Names as the user said them, trimmed and de-duplicated."""
    out: list[str] = []
    for name in raw or []:
        name = " ".join(str(name).split())[:_MAX_COMPANY_CHARS]
        if name and name.lower() not in {o.lower() for o in out}:
            out.append(name)
    return out[:_MAX_TICKERS]


def _default_verdict(reason: str) -> dict[str, Any]:
    """Allow the turn through to finance_qa — used for empty or unclassifiable input."""
    return {
        "is_on_topic": True,
        "next_agent": AgentType.FINANCE_QA,
        "current_agent": AgentType.ROUTER,
        "needs_macro": None,
        "router_reasoning": reason,
    }


def classify_node(state: FinnieState) -> dict[str, Any]:
    """LangGraph node: scope gate + routing in a single structured LLM call."""
    settings = get_settings()
    query = _extract_query(state)

    if not query.strip():
        return _default_verdict("No user message found")

    # 1) Fast-path blocklist — reject obvious NSFW/disallowed without an LLM call.
    if settings.guardrail.enabled:
        matched = _matches_blocklist(query)
        if matched:
            return _reject(f"matched blocklist term '{matched}'")

    # 2) "Stop." / "wait" / "never mind": acknowledge, no model call, no agent.
    if is_acknowledgement(query):
        return _acknowledge()

    # 3) Single structured call, with the previous turn for follow-ups. Any
    #    failure -> fail closed (or route to finance_qa when the scope gate is
    #    switched off entirely).
    try:
        context = conversation_context(*previous_turn(state.messages))
        verdict = classify_query(query, context=context)

        if settings.guardrail.enabled and not verdict.on_topic:
            return _reject(verdict.reason or "classified off-topic")

        next_agent = _AGENT_MAP.get(verdict.agent, AgentType.FINANCE_QA)
        logger.info(
            "classify_decision",
            on_topic=True,
            agent=next_agent.value,
            needs_macro=verdict.needs_macro,
            tickers=_clean_tickers(verdict.tickers),
            companies=len(verdict.companies),
            follow_up=verdict.follow_up,
            with_context=bool(context),
            reason=verdict.reason,
        )
        return {
            "is_on_topic": True,
            "next_agent": next_agent,
            "current_agent": AgentType.ROUTER,
            "needs_macro": verdict.needs_macro,
            "query_tickers": _clean_tickers(verdict.tickers),
            "query_companies": _clean_companies(verdict.companies),
            # Only meaningful with context: without it there is nothing to follow.
            "follow_up": bool(context) and verdict.follow_up,
            "router_reasoning": verdict.reason,
        }

    except Exception as exc:  # noqa: BLE001 — fail closed on any error
        logger.error("classify_error", error=str(exc))
        if settings.guardrail.enabled:
            return _reject(f"classifier error: {exc}")
        # Scope gate disabled: there is nothing to fail closed to, so answer
        # generally rather than dropping the turn.
        return _default_verdict(f"Classifier error: {exc}; defaulting to finance_qa")


def route_after_classify(state: FinnieState) -> str:
    """Conditional edge: 'rejected' or 'acknowledged' -> END, otherwise 'allowed' -> the agent."""
    if state.is_on_topic is False:
        return "rejected"
    if state.next_agent == AgentType.ACKNOWLEDGE:
        return "acknowledged"
    return "allowed"
