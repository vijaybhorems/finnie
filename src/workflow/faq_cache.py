"""FAQ cache nodes — serve repeat questions without any LLM call.

``faq_cache_node`` runs first in the graph. On a hit the turn ends immediately:
no classifier, no agent, no model call. ``faq_cache_write_node`` runs after an
agent and decides whether that answer is safe to reuse.

Only knowledge-grounded answers are cached. An answer that consumed live market
or macro data is never written, because serving yesterday's price or rate as
today's fact is worse than paying for a fresh call.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.core.config import get_settings
from src.core.state import AgentType, FinnieState
from src.utils.logger import get_logger
from src.utils.semantic_cache import get_faq_cache
from src.workflow.guardrail import _extract_query, _matches_blocklist

logger = get_logger(__name__)

# Only these agents produce answers grounded purely in the knowledge base.
_CACHEABLE_AGENTS = {AgentType.FINANCE_QA, AgentType.TAX_EDUCATION}


def _used_live_data(state: FinnieState) -> bool:
    """True when the turn pulled market/news data that dates the answer."""
    data = state.financial_data
    return bool(
        data.tickers
        or data.price_data
        or data.metrics
        or data.news_headlines
    )


def faq_cache_node(state: FinnieState) -> dict[str, Any]:
    """LangGraph node: serve a cached answer when one matches closely enough."""
    settings = get_settings()
    if not settings.fast_path.faq_cache.enabled:
        return {"cache_hit": False}

    query = _extract_query(state)
    if not query.strip():
        return {"cache_hit": False}

    # A blocklisted query must never be answered from cache — the guardrail
    # still owns that decision.
    if settings.guardrail.enabled and _matches_blocklist(query):
        return {"cache_hit": False}

    entry = get_faq_cache().get(query)
    if not entry:
        logger.info("faq_cache_miss")
        return {"cache_hit": False}

    agent = entry.get("agent", AgentType.FINANCE_QA.value)
    logger.info(
        "faq_cache_hit",
        match_type=entry.get("match_type"),
        similarity=round(float(entry.get("similarity", 0.0)), 4),
        agent=agent,
    )
    answer = entry["answer"]
    return {
        "cache_hit": True,
        "is_on_topic": True,
        "final_response": answer,
        "messages": [AIMessage(content=answer, name=agent)],
        "next_agent": AgentType(agent) if agent in AgentType._value2member_map_ else AgentType.FINANCE_QA,
        "router_reasoning": f"Served from FAQ cache ({entry.get('match_type')})",
    }


def route_after_faq_cache(state: FinnieState) -> str:
    """Conditional edge: 'hit' -> END, 'miss' -> the classifier."""
    return "hit" if state.cache_hit else "miss"


def faq_cache_write_node(state: FinnieState) -> dict[str, Any]:
    """LangGraph node: store this turn's answer when it is safe to reuse."""
    settings = get_settings()
    if not settings.fast_path.faq_cache.enabled or state.cache_hit:
        return {}

    if state.next_agent not in _CACHEABLE_AGENTS:
        return {}
    if state.needs_macro is not False:
        # None (legacy path) or True (macro used) — either way the answer may be dated.
        return {}
    if _used_live_data(state):
        return {}
    if state.user_profile.memories:
        # The cache is shared by every user: an answer shaped by this user's
        # saved facts must never be served to someone else.
        return {}

    query = _extract_query(state)
    response = state.final_response
    if not query.strip() or not response.strip():
        return {}

    get_faq_cache().set(query, response, state.next_agent.value)
    return {}
