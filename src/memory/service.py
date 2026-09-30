"""Remember facts from a chat turn — off the request path."""
from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from langgraph.store.base import BaseStore

from src.core.config import get_settings
from src.memory.extractor import extract_facts, is_sensitive
from src.persistence.user_data import UserData
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Small pool: extraction is one short model call per qualifying turn.
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="finnie-memory")
# One extraction at a time per user, so two quick messages can't both miss the
# duplicate check and save the same fact twice (within this process).
_USER_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def remember_turn(user_id: str, message: str, store: Optional[BaseStore] = None) -> list[str]:
    """Extract facts from the user's message and save the ones that qualify.

    Returns the saved facts' text. Respects the global flag and the user's own
    memory switch; drops low-confidence and sensitive facts. Never raises.
    """
    settings = get_settings().memory
    if not settings.enabled:
        return []
    try:
        user_data = UserData(user_id, store)
        if not user_data.memory_enabled():
            return []

        saved: list[str] = []
        with _USER_LOCKS[user_id]:
            for fact in extract_facts(message)[: settings.max_facts_per_turn]:
                if fact.confidence < settings.min_confidence:
                    continue
                if is_sensitive(fact.fact):
                    logger.warning("memory_fact_rejected_sensitive", category=fact.category)
                    continue
                user_data.remember(
                    fact.fact.strip(),
                    fact.category,
                    fact.confidence,
                    fact.expires_on.isoformat() if fact.expires_on else None,
                )
                saved.append(fact.fact.strip())
        if saved:
            logger.info("memory_saved", count=len(saved))
        return saved
    except Exception as exc:  # noqa: BLE001 — never let memory break a chat
        logger.error("memory_save_failed", error=str(exc), error_type=type(exc).__name__)
        return []


def schedule_remember_turn(user_id: str, message: str) -> Future:
    """Run remember_turn in the background; the chat answer never waits on it."""
    return _EXECUTOR.submit(remember_turn, user_id, message)
