"""In-process circuit breaker for downstream API calls."""
from __future__ import annotations

import time
from enum import Enum

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-provider in-process circuit breaker (CLOSED/OPEN/HALF_OPEN)."""

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        recovery_timeout: int,
        success_threshold: int = 1,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow(self) -> bool:
        """Return True if a call may proceed.

        Transitions OPEN -> HALF_OPEN once the recovery timeout has elapsed.
        """
        if self._state is CircuitState.OPEN:
            if time.time() - self._opened_at >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("circuit_half_open", provider=self.name)
                return True
            return False
        return True

    def record_success(self) -> None:
        """Reset failures; close the breaker from HALF_OPEN after enough successes."""
        if self._state is CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info("circuit_closed", provider=self.name)
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Increment failures; open the breaker at threshold or on a HALF_OPEN failure."""
        if self._state is CircuitState.HALF_OPEN:
            self._trip()
            return
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        self._success_count = 0
        logger.warning("circuit_open", provider=self.name, failure_count=self._failure_count)


_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    """Return the process-wide circuit breaker for `name`, built from settings."""
    breaker = _breakers.get(name)
    if breaker is None:
        cb = get_settings().circuit_breaker
        breaker = CircuitBreaker(
            name=name,
            failure_threshold=cb.failure_threshold,
            recovery_timeout=cb.recovery_timeout_seconds,
            success_threshold=cb.success_threshold,
        )
        _breakers[name] = breaker
    return breaker


def reset_breakers() -> None:
    """Clear the breaker registry (primarily for tests)."""
    _breakers.clear()
