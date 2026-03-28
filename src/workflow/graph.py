"""LangGraph workflow — wires router + 6 agents into a state graph."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.goal_planning_agent import GoalPlanningAgent
from src.agents.market_analysis_agent import MarketAnalysisAgent
from src.agents.news_synthesizer_agent import NewsSynthesizerAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.tax_education_agent import TaxEducationAgent
from src.core.state import AgentType, FinnieState
from src.utils.logger import get_logger
from src.workflow.router import router_node

logger = get_logger(__name__)

# ── Agent singletons (lazy-initialised per process) ─────────────────────────
_AGENTS: dict[AgentType, Any] = {}


def _get_agent(agent_type: AgentType):
    if agent_type not in _AGENTS:
        factories = {
            AgentType.FINANCE_QA: FinanceQAAgent,
            AgentType.PORTFOLIO: PortfolioAgent,
            AgentType.MARKET_ANALYSIS: MarketAnalysisAgent,
            AgentType.GOAL_PLANNING: GoalPlanningAgent,
            AgentType.NEWS_SYNTHESIZER: NewsSynthesizerAgent,
            AgentType.TAX_EDUCATION: TaxEducationAgent,
        }
        _AGENTS[agent_type] = factories[agent_type]()
    return _AGENTS[agent_type]


# ── Node wrappers ────────────────────────────────────────────────────────────

def finance_qa_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.FINANCE_QA).run(state)


def portfolio_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.PORTFOLIO).run(state)


def market_analysis_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.MARKET_ANALYSIS).run(state)


def goal_planning_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.GOAL_PLANNING).run(state)


def news_synthesizer_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.NEWS_SYNTHESIZER).run(state)


def tax_education_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.TAX_EDUCATION).run(state)


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_to_agent(state: FinnieState) -> str:
    """Conditional edge: maps next_agent enum → node name."""
    mapping = {
        AgentType.FINANCE_QA: "finance_qa",
        AgentType.PORTFOLIO: "portfolio",
        AgentType.MARKET_ANALYSIS: "market_analysis",
        AgentType.GOAL_PLANNING: "goal_planning",
        AgentType.NEWS_SYNTHESIZER: "news_synthesizer",
        AgentType.TAX_EDUCATION: "tax_education",
    }
    destination = mapping.get(state.next_agent, "finance_qa")
    logger.info("routing_to_agent", destination=destination)
    return destination


# ── Graph construction ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_graph():
    """Build and compile the LangGraph workflow. Cached per process."""
    graph = StateGraph(FinnieState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("finance_qa", finance_qa_node)
    graph.add_node("portfolio", portfolio_node)
    graph.add_node("market_analysis", market_analysis_node)
    graph.add_node("goal_planning", goal_planning_node)
    graph.add_node("news_synthesizer", news_synthesizer_node)
    graph.add_node("tax_education", tax_education_node)

    # Start → router
    graph.add_edge(START, "router")

    # Router → conditional dispatch
    graph.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "finance_qa": "finance_qa",
            "portfolio": "portfolio",
            "market_analysis": "market_analysis",
            "goal_planning": "goal_planning",
            "news_synthesizer": "news_synthesizer",
            "tax_education": "tax_education",
        },
    )

    # All agent nodes → END
    for node in ["finance_qa", "portfolio", "market_analysis", "goal_planning", "news_synthesizer", "tax_education"]:
        graph.add_edge(node, END)

    compiled = graph.compile()
    logger.info("langgraph_workflow_compiled")
    return compiled


def run_workflow(
    user_message: str,
    conversation_history: list | None = None,
    user_profile: dict | None = None,
) -> dict[str, Any]:
    """
    Main entry point for running a single turn through the workflow.

    Returns a dict with keys: final_response, agent_used, financial_data, rag_context.
    """
    from langchain_core.messages import HumanMessage

    from src.core.state import FinancialData, UserProfile

    graph = build_graph()
    history = list(conversation_history or [])
    history.append(HumanMessage(content=user_message))

    profile = UserProfile(**(user_profile or {})) if user_profile else UserProfile()

    initial_state = FinnieState(
        messages=history,
        user_profile=profile,
        financial_data=FinancialData(),
    )

    try:
        result = graph.invoke(initial_state)
        return {
            "final_response": result.get("final_response", ""),
            "agent_used": result.get("next_agent", AgentType.FINANCE_QA).value if hasattr(result.get("next_agent"), "value") else str(result.get("next_agent", "")),
            "router_reasoning": result.get("router_reasoning", ""),
            "financial_data": result.get("financial_data", FinancialData()).model_dump(),
            "rag_context": result.get("rag_context", []),
            "messages": result.get("messages", []),
        }
    except Exception as exc:
        logger.error("workflow_error", error=str(exc))
        return {
            "final_response": f"I encountered an error: {exc}. Please try again.",
            "agent_used": "error",
            "router_reasoning": "",
            "financial_data": {},
            "rag_context": [],
            "messages": history,
        }
