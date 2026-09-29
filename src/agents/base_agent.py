"""Base class shared by all Finnie agents."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
from langchain_core.messages import BaseMessage, SystemMessage

from src.agents.prompts import load_prompt, shared_system_prompt
from src.core.config import get_settings
from src.core.llm import get_llm, message_text, token_usage
from src.core.state import FinnieState
from src.utils.logger import get_logger

# Exceptions that indicate a transient network issue and are safe to retry.
_RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)
_MAX_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 1  # exponential: 1s, 2s, 4s

_DISCLAIMER = (
    "\n\n---\n*Disclaimer: This is educational information only, not financial advice. "
    "Consult a registered financial advisor before making investment decisions.*"
)

# Agents already warned that their prompt is not being cached (warn once each).
_CACHE_INACTIVE_WARNED: set[str] = set()


def _cache_control() -> Optional[dict[str, str]]:
    """The cache_control marker for stable blocks, or None when caching is off."""
    caching = get_settings().llm.prompt_caching
    if not caching.enabled:
        return None
    control = {"type": "ephemeral"}
    if caching.ttl == "1h":
        control["ttl"] = "1h"
    return control


def _text_block(text: str, cache_control: Optional[dict[str, str]] = None) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache_control:
        block["cache_control"] = cache_control
    return block


class BaseAgent(ABC):
    """Abstract base for all Finnie agents."""

    name: str = "Base Agent"
    description: str = "Base financial agent"
    # Task instructions: src/agents/prompts/<prompt_name>.md
    prompt_name: str = ""

    def __init__(self) -> None:
        self._llm = get_llm()
        self._logger = get_logger(self.__class__.__name__)

    # ── Prompt assembly ──────────────────────────────────────────────────────
    #
    # Prompt caching is a prefix match, so the system prompt is ordered from
    # most to least stable (see src/agents/prompts/__init__.py):
    #
    #   1. shared_system_prompt()  identical for all agents          [cached]
    #   2. role_prompt             this agent's role and task        [cached]
    #   3. request context         profile, retrieved text, live data
    #
    # Blocks 1 and 2 must never contain anything that varies per request —
    # a timestamp or a user id there would silently turn every call into a
    # cache write. Per-request content belongs in the `context` argument of
    # _invoke_llm, which lands in block 3.

    @property
    def role_prompt(self) -> str:
        """This agent's stable instructions: role, task and static reference data."""
        parts = [f"# Your role: {self.name}\n\n{self.description}"]
        if self.prompt_name:
            parts.append(load_prompt(self.prompt_name))
        reference = self._static_reference()
        if reference:
            parts.append(reference)
        return "\n\n".join(parts)

    def _static_reference(self) -> str:
        """Reference data that is the same on every request (cached with the role)."""
        return ""

    @property
    def system_prompt(self) -> str:
        """The full stable system text (blocks 1 + 2), for inspection and token counts."""
        return f"{shared_system_prompt()}\n\n{self.role_prompt}"

    def _build_messages(self, state: FinnieState, context: str = "") -> list[BaseMessage]:
        """System prompt as content blocks, followed by the conversation.

        Built from message objects rather than a ChatPromptTemplate, so text
        containing literal braces (JSON in articles or data) needs no escaping.
        """
        cache = _cache_control()
        blocks = [
            _text_block(shared_system_prompt(), cache),
            _text_block(self.role_prompt, cache),
        ]
        # The API rejects empty text blocks, so an agent with no per-request
        # context sends only the two stable blocks.
        if context.strip():
            blocks.append(_text_block(f"# Context for this request\n\n{context.strip()}"))
        return [SystemMessage(content=blocks), *state.messages]

    def _get_user_context_str(self, state: FinnieState) -> str:
        profile = state.user_profile
        return (
            f"User profile: knowledge_level={profile.knowledge_level}, "
            f"risk_tolerance={profile.risk_tolerance}, "
            f"investment_horizon={profile.investment_horizon}"
        )

    def _add_disclaimer(self, text: str) -> str:
        return text + _DISCLAIMER

    @abstractmethod
    def run(self, state: FinnieState) -> dict[str, Any]:
        """Process the state and return updated state dict."""
        ...

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _check_cache_active(self, usage: dict[str, int]) -> None:
        """Warn once per agent when caching is on but the API cached nothing.

        Neither a read nor a write means the stable prefix was not cached at
        all — usually because it is below the model's minimum cacheable length
        (1,024 tokens on Claude Sonnet 5). The request still succeeds, so this
        log is the only signal.
        """
        if not usage or _cache_control() is None or self.name in _CACHE_INACTIVE_WARNED:
            return
        if usage["cache_read_input_tokens"] == 0 and usage["cache_creation_input_tokens"] == 0:
            _CACHE_INACTIVE_WARNED.add(self.name)
            self._logger.warning(
                "prompt_cache_inactive",
                agent=self.name,
                input_tokens=usage["input_tokens"],
                hint="stable system prompt is likely below the model's minimum cacheable length",
            )

    def _invoke_llm(self, state: FinnieState, context: str = "") -> str:
        """Invoke the LLM with retry on transient connection errors.

        `context` is this request's volatile content (user profile, retrieved
        passages, live data); it is placed after the cached prompt blocks.

        Metrics emitted (structured logs → GCP log-based metrics):
          llm_call_success  — successful invocation (fields: agent, attempt, latency_ms,
                              input_tokens, cache_read_input_tokens,
                              cache_creation_input_tokens, output_tokens)
          llm_call_retry    — retrying after a connection error (fields: agent, attempt, retry_in_seconds, error)
          llm_call_failed   — all retries exhausted or non-retryable error (fields: agent, attempt, error_type, error)
          prompt_cache_inactive — caching enabled but nothing cached (once per agent)
        """
        messages = self._build_messages(state, context)

        for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
            t0 = time.monotonic()
            try:
                response = self._llm.invoke(messages)
                latency_ms = int((time.monotonic() - t0) * 1000)
                usage = token_usage(response)
                self._logger.info(
                    "llm_call_success",
                    agent=self.name,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    **usage,
                )
                self._check_cache_active(usage)
                return message_text(response)

            except _RETRYABLE_EXCEPTIONS as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                if attempt < _MAX_RETRY_ATTEMPTS:
                    retry_in = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))  # 1s, 2s
                    self._logger.warning(
                        "llm_call_retry",
                        agent=self.name,
                        attempt=attempt,
                        max_attempts=_MAX_RETRY_ATTEMPTS,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        retry_in_seconds=retry_in,
                        latency_ms=latency_ms,
                    )
                    time.sleep(retry_in)
                else:
                    self._logger.error(
                        "llm_call_failed",
                        agent=self.name,
                        attempt=attempt,
                        error=str(exc),
                        error_type="connection_error",
                        latency_ms=latency_ms,
                    )
                    return (
                        "I'm having trouble connecting to my AI service right now. "
                        "Please try again in a moment."
                    )

            except Exception as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                self._logger.error(
                    "llm_call_failed",
                    agent=self.name,
                    attempt=attempt,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    latency_ms=latency_ms,
                )
                return f"I encountered an error processing your request: {exc}"

        # Should never reach here, but satisfies the type checker.
        return "Unexpected error. Please try again."
