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
