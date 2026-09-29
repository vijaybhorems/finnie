"""Classify node — one structured LLM call that replaces guardrail + router.

The legacy pipeline spent two serial LLM calls before an agent started: a
finance/NSFW gate, then an agent selector. Both asked the model for JSON and
parsed it out of free text with a regex. This node asks once, with structured
output, and returns the scope verdict, the destination agent and whether live
macro data is relevant.

Safety semantics are unchanged and still fail CLOSED: the blocklist fast path
rejects without an LLM call, and any error, malformed verdict or off-topic
result routes straight to END with the canned refusal.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.llm import get_llm
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


class Verdict(BaseModel):
    """Structured classifier output — scope gate and routing in one shot."""

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
3. needs_macro — would today's macro data (fed funds rate, CPI, GDP, unemployment,
   treasury yields) materially change the answer? Definitions and conceptual
   explanations do not need it.

Examples:
- "What is a P/E ratio?" -> on_topic, finance_qa, needs_macro=false
- "How do current interest rates affect bond prices?" -> on_topic, finance_qa, needs_macro=true
- "Analyze my portfolio: AAPL 10 shares @ $150" -> on_topic, portfolio, needs_macro=false
- "What is Apple stock doing today?" -> on_topic, market_analysis, needs_macro=false
- "How much do I need to retire at 55?" -> on_topic, goal_planning, needs_macro=true
- "How does a Roth IRA work?" -> on_topic, tax_education, needs_macro=false
- "Give me a lasagna recipe" -> off_topic
"""


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

    # 2) Single structured call. Any failure -> fail closed (or route to
    #    finance_qa when the scope gate is switched off entirely).
    try:
        verdict = get_llm(streaming=False).with_structured_output(Verdict).invoke(
            [
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": f"Classify this query: {query}"},
            ]
        )
        if verdict is None:
            raise ValueError("classifier returned no verdict")
        if isinstance(verdict, dict):
            verdict = Verdict(**verdict)

        if settings.guardrail.enabled and not verdict.on_topic:
            return _reject(verdict.reason or "classified off-topic")

        next_agent = _AGENT_MAP.get(verdict.agent, AgentType.FINANCE_QA)
        logger.info(
            "classify_decision",
            on_topic=True,
            agent=next_agent.value,
            needs_macro=verdict.needs_macro,
            reason=verdict.reason,
        )
        return {
            "is_on_topic": True,
            "next_agent": next_agent,
            "current_agent": AgentType.ROUTER,
            "needs_macro": verdict.needs_macro,
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
    """Conditional edge: 'rejected' -> END, otherwise 'allowed' -> the agent."""
    return "rejected" if state.is_on_topic is False else "allowed"
