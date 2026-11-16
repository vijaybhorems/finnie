"""Memories from voice turns: which turns to learn from, and waiting for the saves.

Pure (no LiveKit), so CI tests it. Voice uses the chat page's memory service
(src/memory/service.py) with two extra guards, because a misheard sentence can
become a wrong "fact" about the user (design doc, Impact item 7):

- a turn is skipped when Deepgram was unsure of the transcript
  (`voice.memory_min_transcript_confidence`), as well as for refusals and errors;
- a fact must clear `voice.memory_min_confidence`, higher than chat's.

Saves run in the background like chat's. When a call ends, `MemorySaves.wait()`
gives the ones still running up to `voice.memory_wait_seconds`; a save that
misses it is logged, not retried (the process is about to exit).
"""
from __future__ import annotations

import concurrent.futures
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Turns with nothing worth learning from: refused (off topic) or failed.
_NOT_LEARNED_FROM = {"out_of_scope", "error"}


def should_remember(agent_used: Optional[str], transcript_confidence: Optional[float], voice: Any) -> bool:
    """True when a finished voice turn may be mined for facts about the user."""
    if not agent_used or agent_used in _NOT_LEARNED_FROM:
        return False
    if transcript_confidence is None:  # the speech-to-text vendor gave none
        return True
    return transcript_confidence >= voice.memory_min_transcript_confidence


class MemorySaves:
    """The memory saves a call started, so the call can wait for them as it ends."""

    def __init__(self, schedule: Optional[Callable[..., Future]] = None) -> None:
        if schedule is None:
            from src.memory.service import schedule_remember_turn

            schedule = schedule_remember_turn
        self._schedule = schedule
        self._futures: list[Future] = []
        self._lock = threading.Lock()  # added from the graph's thread, waited on from the call's

    def add(self, user_id: str, message: str, min_confidence: float) -> None:
        future = self._schedule(user_id, message, min_confidence=min_confidence)
        with self._lock:
            self._futures.append(future)

    def wait(self, timeout_s: float) -> int:
        """Block until the saves finish or `timeout_s` passes. Returns how many missed."""
        with self._lock:
            pending = [f for f in self._futures if not f.done()]
            self._futures = []
        if not pending:
            return 0
        _, missed = concurrent.futures.wait(pending, timeout=timeout_s)
        if missed:
            logger.warning("voice_memory_save_missed", count=len(missed), waited_s=timeout_s)
        return len(missed)
