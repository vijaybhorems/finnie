"""Circuit breakers for the voice vendors (Deepgram, ElevenLabs), shared between calls.

LiveKit runs every call in its own short-lived process, so the in-process
breakers in src/utils/circuit_breaker.py would forget each failure when its
call ends. These keep their state in the shared cache instead (Redis, else
in-process memory: without Redis a breaker only sees failures inside one call).

Rules, using the `circuit_breaker` settings:
- `failure_threshold` failures in a row open a vendor's breaker;
- while it is open, a new call that needs the vendor ends straight away with
  the spoken "unavailable" line instead of failing mid-call;
- after `recovery_timeout_seconds` the next call is let through as a probe:
  a success closes the breaker, another failure re-opens it.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEEPGRAM = "deepgram"      # speech to text, and the backup voice
ELEVENLABS = "elevenlabs"  # the main voice


class VendorHealth:
    def __init__(self, store: Any = None, clock: Callable[[], float] = time.time) -> None:
        if store is None:
            from src.utils.cache import get_cache

            store = get_cache()
        cb = get_settings().circuit_breaker
        self._store, self._clock = store, clock
        self._threshold, self._recovery_s = cb.failure_threshold, cb.recovery_timeout_seconds

    def _key(self, vendor: str) -> str:
        return f"finnie:voice:health:{vendor}"

    def _state(self, vendor: str) -> dict[str, Any]:
        value = self._store.get(self._key(vendor))
        return value if isinstance(value, dict) else {}

    def is_open(self, vendor: str) -> bool:
        """True while new calls should not try `vendor`."""
        opened_at: Optional[float] = self._state(vendor).get("opened_at")
        return opened_at is not None and self._clock() - opened_at < self._recovery_s

    def failure(self, vendor: str) -> None:
        state = self._state(vendor)
        failures = int(state.get("failures", 0)) + 1
        probe_failed = state.get("opened_at") is not None  # open or half-open: re-open
        opened_at = self._clock() if failures >= self._threshold or probe_failed else None
        if opened_at is not None and not self.is_open(vendor):
            logger.warning("voice_vendor_circuit_open", vendor=vendor, failures=failures)
        # Kept long enough to outlive the open period, then forgotten.
        self._store.set(self._key(vendor), {"failures": failures, "opened_at": opened_at},
                        ttl=max(3600, self._recovery_s * 10))

    def success(self, vendor: str) -> None:
        if self._state(vendor):
            if self._state(vendor).get("opened_at") is not None:
                logger.info("voice_vendor_circuit_closed", vendor=vendor)
            self._store.delete(self._key(vendor))
