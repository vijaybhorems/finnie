"""Base class shared by all Finnie agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm
from src.core.state import FinnieState
from src.utils.logger import get_logger

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
        """Helper to invoke the LLM with conversation history."""
        prompt = self._build_prompt(additional_system)
        chain = prompt | self._llm
        try:
            response = chain.invoke({"messages": state.messages})
            return response.content
        except Exception as exc:
            self._logger.error("llm_invocation_error", agent=self.name, error=str(exc))
            return f"I encountered an error processing your request: {exc}"
