"""Router node — classifies user intent and routes to the appropriate agent."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.core.llm import get_llm
from src.core.state import AgentType, FinnieState
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ROUTER_SYSTEM = """You are the router for Finnie, an AI finance assistant. Your ONLY job is to classify
user queries and route them to the most appropriate specialized agent.

Available agents:
- finance_qa: General financial education (concepts, definitions, how investing works, macroeconomics, market fundamentals)
- portfolio: Portfolio analysis (analyzing holdings, P/E ratios, diversification, performance metrics, risk assessment)
- market_analysis: Real-time market data (stock prices, technical indicators, sector performance, market conditions)
- goal_planning: Financial goal setting (retirement planning, savings goals, projections, risk-based planning, FIRE)
- news_synthesizer: Financial news (current events, earnings, economic news, market-moving events, SEC filings)
- tax_education: Tax concepts (capital gains, 401k, IRA, Roth, HSA, tax-loss harvesting, account types)

Respond with ONLY a JSON object in this exact format:
{
  "agent": "<agent_name>",
  "reasoning": "<one sentence explaining why>"
}

Examples:
- "What is a P/E ratio?" → finance_qa (conceptual question)
- "Analyze my portfolio: AAPL 10 shares @ $150" → portfolio
- "What is Apple stock doing today?" → market_analysis
- "How much do I need to retire at 55?" → goal_planning
- "What happened to the market today?" → news_synthesizer
- "How does a Roth IRA work?" → tax_education
- "Should I do tax-loss harvesting?" → tax_education
- "What's RSI showing for Tesla?" → market_analysis
"""


def router_node(state: FinnieState) -> dict[str, Any]:
    """LangGraph node that classifies intent and sets next_agent."""
    llm = get_llm()

    user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
    if not user_messages:
        return {"next_agent": AgentType.FINANCE_QA, "router_reasoning": "No user message found"}

    query = user_messages[-1].content

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=f"Route this query: {query}"),
        ])

        import json
        import re
        raw = response.content.strip()
        # Extract JSON even if wrapped in markdown code blocks
        json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            agent_str = parsed.get("agent", "finance_qa").lower()
            reasoning = parsed.get("reasoning", "")
        else:
            agent_str = "finance_qa"
            reasoning = "Fallback to finance_qa (JSON parse failed)"

        # Map string to enum
        agent_map = {
            "finance_qa": AgentType.FINANCE_QA,
            "portfolio": AgentType.PORTFOLIO,
            "market_analysis": AgentType.MARKET_ANALYSIS,
            "goal_planning": AgentType.GOAL_PLANNING,
            "news_synthesizer": AgentType.NEWS_SYNTHESIZER,
            "tax_education": AgentType.TAX_EDUCATION,
        }
        next_agent = agent_map.get(agent_str, AgentType.FINANCE_QA)
        logger.info("router_decision", agent=next_agent.value, reasoning=reasoning)

        return {
            "next_agent": next_agent,
            "router_reasoning": reasoning,
            "current_agent": AgentType.ROUTER,
        }

    except Exception as exc:
        logger.error("router_error", error=str(exc))
        return {
            "next_agent": AgentType.FINANCE_QA,
            "router_reasoning": f"Router error: {exc}; defaulting to finance_qa",
        }
