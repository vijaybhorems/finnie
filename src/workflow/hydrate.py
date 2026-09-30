"""Hydrate node — load the signed-in user's saved data into this turn's state.

Runs after an FAQ-cache miss and before classification, so agents see the
profile, holdings and saved plan the user set on the other tabs. A cache hit
skips it: a cached answer uses no personal context.

The user id is read from the run config, which run_workflow fills from the
server-side authenticated identity — never from state, which carries data the
page supplied. Fails open: if the store is unreachable the turn proceeds with
the profile it was given rather than failing the user's question.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from src.core.state import FinnieState
from src.persistence.user_data import UserData
from src.utils.logger import get_logger
from src.workflow.guardrail import _extract_query

logger = get_logger(__name__)


def hydrate_node(
    state: FinnieState,
    config: RunnableConfig,
    *,
    # LangGraph injects the compiled store only for the annotation spellings
    # "BaseStore" and "Optional[BaseStore]". "BaseStore | None" is silently not
    # injected, which would make this node a no-op with no error anywhere.
    store: Optional[BaseStore] = None,
) -> dict[str, Any]:
    """LangGraph node: replace state.user_profile with the user's saved data."""
    user_id = (config.get("configurable") or {}).get("user_id")
    if not user_id or store is None:
        return {}
    try:
        user_data = UserData(user_id, store)
        profile = user_data.to_user_profile()
    except Exception as exc:  # noqa: BLE001 — answer without personal context rather than fail
        logger.error("hydrate_failed", error=str(exc), error_type=type(exc).__name__)
        return {}

    # Memories are optional context: a failed recall must not drop the profile.
    try:
        profile.memories = user_data.relevant_memories(_extract_query(state))
    except Exception as exc:  # noqa: BLE001
        logger.error("memory_recall_failed", error=str(exc), error_type=type(exc).__name__)

    logger.info(
        "hydrate_loaded",
        holdings=len(profile.portfolio),
        has_plan=bool(profile.goals),
        memories=len(profile.memories),
    )
    return {"user_profile": profile}
