"""Base class shared by all Finnie agents."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm
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


class BaseAgent(ABC):
    """Abstract base for all Finnie agents."""

    name: str = "Base Agent"
    description: str = "Base financial agent"

    def __init__(self) -> None:
        self._llm = get_llm()
        self._logger = get_logger(self.__class__.__name__)

    @property
    def system_prompt(self) -> str:
        return (
            f"You are {self.name}, a specialized AI assistant that is part of Finnie, "
            f"an AI-powered personal finance education platform. {self.description}\n\n"
            "IMPORTANT GUIDELINES:\n"
            "- You provide financial EDUCATION, not personalized financial advice\n"
            "- Always include appropriate disclaimers when discussing specific investments\n"
            "- Be clear, jargon-free, and calibrate explanations to the user's knowledge level\n"
            "- Cite your sources when referencing specific data\n"
            "- If uncertain about a fact, say so explicitly\n"
            "- Never recommend specific securities as 'buys' or 'sells'"
        )

    def _build_prompt(self, additional_system: str = "") -> ChatPromptTemplate:
        system = self.system_prompt
        if additional_system:
            system = f"{system}\n\n{additional_system}"
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("placeholder", "{messages}"),
        ])

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

    def _invoke_llm(self, state: FinnieState, additional_system: str = "") -> str:
        """Helper to invoke the LLM with retry on transient connection errors.

        Metrics emitted (structured logs → GCP log-based metrics):
          llm_call_success  — successful invocation (fields: agent, attempt, latency_ms)
          llm_call_retry    — retrying after a connection error (fields: agent, attempt, retry_in_seconds, error)
          llm_call_failed   — all retries exhausted or non-retryable error (fields: agent, attempt, error_type, error)
        """
        prompt = self._build_prompt(additional_system)
        chain = prompt | self._llm

        for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
            t0 = time.monotonic()
            try:
                response = chain.invoke({"messages": state.messages})
                latency_ms = int((time.monotonic() - t0) * 1000)
                self._logger.info(
                    "llm_call_success",
                    agent=self.name,
                    attempt=attempt,
                    latency_ms=latency_ms,
                )
                return response.content

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
