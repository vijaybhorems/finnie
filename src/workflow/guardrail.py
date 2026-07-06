"""Guardrail node — finance/NSFW gate that runs before the router.

Classifies each incoming query as in-scope (finance) vs. off-topic/NSFW. On
rejection it sets a canned refusal as ``final_response`` and short-circuits the
graph to END, bypassing the router and all agents.

Design: fail CLOSED. Any classifier/LLM error, malformed output, or blocklist
hit results in rejection — the safety path never silently forwards a
questionable query to an agent.
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.core.config import get_settings
from src.core.llm import get_llm
from src.core.state import AgentType, FinnieState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Obvious NSFW / disallowed terms handled without an LLM call (fast-path reject).
# Word-boundary matched, case-insensitive; keep conservative to avoid false positives.
_NSFW_TERMS: set[str] = {
    "porn", "pornography", "nsfw", "sex", "sexual", "nude", "nudes", "naked",
    "erotic", "erotica", "xxx", "blowjob", "handjob", "orgasm", "masturbate",
    "masturbation", "bestiality", "incest", "rape", "fetish", "hentai",
    "kill", "murder", "suicide", "bomb", "explosive", "meth", "cocaine",
    "heroin", "how to make a weapon",
}

_GUARDRAIL_SYSTEM = """You are the safety and scope gate for Finnie, an AI \
finance education assistant. Decide whether a user query is ON-TOPIC.

ON-TOPIC means the query is about personal finance, investing, markets, \
stocks, bonds, portfolios, budgeting, saving, retirement, taxes, economics, \
financial planning, financial news, or related educational concepts. General \
greetings or clarifying meta-questions about Finnie's capabilities are also \
ON-TOPIC.

OFF-TOPIC means anything unrelated to finance (e.g. cooking, coding help, \
relationships, medical/legal advice, general trivia) OR any NSFW, sexual, \
violent, illegal, or otherwise harmful content.

Respond with ONLY a JSON object in this exact format:
{
  "on_topic": true or false,
  "reason": "<one short sentence>"
}
"""


def _extract_query(state: FinnieState) -> str:
    user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
    return str(user_messages[-1].content) if user_messages else ""


def _matches_blocklist(query: str) -> str | None:
    """Return the first blocklisted term found in `query`, else None."""
    lowered = query.lower()
    for term in _NSFW_TERMS:
        # Multi-word phrases: simple substring; single words: word-boundary match.
        if " " in term:
            if term in lowered:
                return term
        elif re.search(rf"\b{re.escape(term)}\b", lowered):
            return term
    return None


def _reject(reason: str) -> dict[str, Any]:
    settings = get_settings()
    logger.warning("guardrail_rejected", reason=reason)
    return {
        "is_on_topic": False,
        "next_agent": AgentType.OUT_OF_SCOPE,
        "current_agent": AgentType.OUT_OF_SCOPE,
        "final_response": settings.guardrail.refusal_message,
        "router_reasoning": f"Blocked by guardrail: {reason}",
    }


def guardrail_node(state: FinnieState) -> dict[str, Any]:
    """LangGraph node: gate finance/NSFW before routing. Fails closed."""
    settings = get_settings()

    # Feature flag: when disabled, treat everything as on-topic (pass through).
    if not settings.guardrail.enabled:
        return {"is_on_topic": True}

    query = _extract_query(state)
    if not query.strip():
        # No content to evaluate — let the router handle the empty case.
        return {"is_on_topic": True}

    # 1) Fast-path blocklist — reject obvious NSFW/disallowed without an LLM call.
    matched = _matches_blocklist(query)
    if matched:
        return _reject(f"matched blocklist term '{matched}'")

    # 2) LLM topic classification. Any failure/ambiguity → reject (fail closed).
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_GUARDRAIL_SYSTEM),
            HumanMessage(content=f"Classify this query: {query}"),
        ])
        raw = str(response.content).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return _reject("classifier returned no parseable verdict")

        parsed = json.loads(json_match.group())
        on_topic = bool(parsed.get("on_topic", False))
        reason = str(parsed.get("reason", ""))

        if on_topic:
            logger.info("guardrail_decision", on_topic=True, reason=reason)
            return {"is_on_topic": True, "router_reasoning": reason}
        return _reject(reason or "classified off-topic")

    except Exception as exc:  # noqa: BLE001 — fail closed on any error
        logger.error("guardrail_error", error=str(exc))
        return _reject(f"classifier error: {exc}")


def route_after_guardrail(state: FinnieState) -> str:
    """Conditional edge: 'rejected' → END, otherwise 'allowed' → router."""
    return "rejected" if state.is_on_topic is False else "allowed"
