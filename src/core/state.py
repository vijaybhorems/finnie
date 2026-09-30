"""LangGraph state definition for the Finnie workflow."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentType(str, Enum):
    FINANCE_QA = "finance_qa"
    PORTFOLIO = "portfolio"
    MARKET_ANALYSIS = "market_analysis"
    GOAL_PLANNING = "goal_planning"
    NEWS_SYNTHESIZER = "news_synthesizer"
    TAX_EDUCATION = "tax_education"
    ROUTER = "router"
    OUT_OF_SCOPE = "out_of_scope"


class UserProfile(BaseModel):
    """Persisted user context across turns."""
    user_id: str = "default"
    risk_tolerance: str = "moderate"  # conservative / moderate / aggressive
    investment_horizon: str = "long"  # short / medium / long
    portfolio: list[dict[str, Any]] = Field(default_factory=list)
    goals: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_level: str = "beginner"  # beginner / intermediate / advanced


class FinancialData(BaseModel):
    """Structured market/financial data attached to a response."""
    tickers: list[str] = Field(default_factory=list)
    price_data: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    news_headlines: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class FinnieState(BaseModel):
    """Global workflow state threaded through every LangGraph node."""

    # Conversation history (LangGraph managed)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Routing
    current_agent: AgentType = AgentType.ROUTER
    next_agent: Optional[AgentType] = None
    router_reasoning: str = ""

    # Guardrail verdict (set by the guardrail/classify node; None until evaluated)
    is_on_topic: Optional[bool] = None

    # Whether the classifier judged live macro data relevant to this query.
    # None = not classified (legacy guardrail+router path) -> agents fetch as before.
    needs_macro: Optional[bool] = None

    # True when the response was served from the semantic FAQ cache (no LLM call).
    cache_hit: bool = False

    # User context
    user_profile: UserProfile = Field(default_factory=UserProfile)

    # Data payloads
    financial_data: FinancialData = Field(default_factory=FinancialData)
    rag_context: list[str] = Field(default_factory=list)

    # Final response assembled by nodes
    final_response: str = ""
    error: Optional[str] = None

    # Iteration guard
    iteration_count: int = 0


def turn_state_reset() -> dict[str, Any]:
    """Fresh values for every per-turn field of FinnieState.

    On a checkpointed thread every field persists into the next turn, and only
    `messages` has a reducer (append). Anything a turn computes must therefore
    be reset at the start of the next one, or it leaks: a previous market
    turn's financial_data would stop the FAQ cache writing, and a stale
    final_response could be returned if a later node fails.

    Built from the model's own defaults so a field added later is reset too.
    """
    defaults = FinnieState()
    return {name: getattr(defaults, name) for name in FinnieState.model_fields if name != "messages"}


def trim_history(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """The most recent `max_messages` messages, starting at a human turn.

    Persistent threads keep the whole conversation; this bounds what each LLM
    call pays for. The Messages API requires the first message to be from the
    user, so a window that would open on an assistant reply starts one later.
    """
    recent = list(messages[-max_messages:]) if 0 < max_messages < len(messages) else list(messages)
    while recent and getattr(recent[0], "type", None) != "human":
        recent.pop(0)
    return recent or list(messages)
