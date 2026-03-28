"""Router accuracy evals.

Tests that the router correctly classifies user intent and routes to the
right agent across a wide range of queries.

Two modes:
  - MOCKED (default): Tests the routing logic with a mock LLM that returns
    deterministic JSON responses.
  - LIVE: Set FINNIE_EVAL_LIVE=1 to test with the real Claude API.

Run with:
    pytest tests/evals/test_router_evals.py -v
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.core.state import AgentType, FinancialData, FinnieState, UserProfile

LIVE_MODE = os.environ.get("FINNIE_EVAL_LIVE", "0") == "1"


def _make_state(query: str) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(),
        financial_data=FinancialData(),
    )


# ---------------------------------------------------------------------------
# Comprehensive routing test cases
# ---------------------------------------------------------------------------
ROUTING_CASES = [
    # Finance Q&A
    ("What is a P/E ratio?", AgentType.FINANCE_QA),
    ("Explain compound interest to me", AgentType.FINANCE_QA),
    ("What is the difference between stocks and bonds?", AgentType.FINANCE_QA),
    ("How does inflation affect my money?", AgentType.FINANCE_QA),
    ("What is an expense ratio?", AgentType.FINANCE_QA),
    ("Define dollar-cost averaging", AgentType.FINANCE_QA),

    # Portfolio
    ("Analyze my portfolio: AAPL 10 shares @ $150, MSFT 5 shares @ $300", AgentType.PORTFOLIO),
    ("How diversified is my portfolio?", AgentType.PORTFOLIO),
    ("What is the beta of my holdings?", AgentType.PORTFOLIO),
    ("Check my portfolio performance", AgentType.PORTFOLIO),

    # Market Analysis
    ("What is Apple stock doing today?", AgentType.MARKET_ANALYSIS),
    ("Show me the RSI for Tesla", AgentType.MARKET_ANALYSIS),
    ("What's the current price of NVDA?", AgentType.MARKET_ANALYSIS),
    ("How are tech stocks performing?", AgentType.MARKET_ANALYSIS),
    ("Is there a golden cross on the S&P 500?", AgentType.MARKET_ANALYSIS),

    # Goal Planning
    ("How much do I need to retire at 55?", AgentType.GOAL_PLANNING),
    ("Can I save $100k in 5 years?", AgentType.GOAL_PLANNING),
    ("I want to buy a house in 3 years, how much should I save?", AgentType.GOAL_PLANNING),
    ("What is the FIRE movement and how do I calculate my number?", AgentType.GOAL_PLANNING),
    ("Plan my retirement savings", AgentType.GOAL_PLANNING),

    # News
    ("What happened in the market today?", AgentType.NEWS_SYNTHESIZER),
    ("Any news about the Fed meeting?", AgentType.NEWS_SYNTHESIZER),
    ("What are the latest earnings reports?", AgentType.NEWS_SYNTHESIZER),
    ("Tell me about recent SEC filings", AgentType.NEWS_SYNTHESIZER),

    # Tax Education
    ("How does a Roth IRA work?", AgentType.TAX_EDUCATION),
    ("What are capital gains taxes?", AgentType.TAX_EDUCATION),
    ("Should I do tax-loss harvesting?", AgentType.TAX_EDUCATION),
    ("What is the 2024 401k contribution limit?", AgentType.TAX_EDUCATION),
    ("How does an HSA work for investing?", AgentType.TAX_EDUCATION),
    ("What's the wash-sale rule?", AgentType.TAX_EDUCATION),
]


# ---------------------------------------------------------------------------
# Mocked router tests
# ---------------------------------------------------------------------------
class _MockRouterLLM:
    """Mock LLM that returns the expected agent routing based on keyword matching."""

    KEYWORD_MAP = {
        "portfolio": "portfolio",
        "holdings": "portfolio",
        "diversified": "portfolio",
        "beta of my": "portfolio",
        "stock doing": "market_analysis",
        "rsi": "market_analysis",
        "current price": "market_analysis",
        "tech stocks": "market_analysis",
        "golden cross": "market_analysis",
        "retire": "goal_planning",
        "save $": "goal_planning",
        "buy a house": "goal_planning",
        "fire movement": "goal_planning",
        "savings": "goal_planning",
        "market today": "news_synthesizer",
        "news": "news_synthesizer",
        "earnings reports": "news_synthesizer",
        "sec filings": "news_synthesizer",
        "fed meeting": "news_synthesizer",
        "roth ira": "tax_education",
        "capital gains": "tax_education",
        "tax-loss": "tax_education",
        "401k": "tax_education",
        "hsa": "tax_education",
        "wash-sale": "tax_education",
    }

    def invoke(self, messages):
        query = ""
        for m in messages:
            content = getattr(m, "content", str(m))
            if "Route this query:" in content:
                query = content.lower()
                break

        agent = "finance_qa"
        for keyword, agent_name in self.KEYWORD_MAP.items():
            if keyword.lower() in query:
                agent = agent_name
                break

        result = json.dumps({"agent": agent, "reasoning": f"Matched keyword for {agent}"})
        return MagicMock(content=result)


class TestRouterAccuracy:
    """Test router accuracy with a deterministic mock LLM."""

    @pytest.mark.parametrize(
        "query,expected_agent",
        ROUTING_CASES,
        ids=[f"{a.value}:{q[:40]}" for q, a in ROUTING_CASES],
    )
    def test_routes_to_correct_agent(self, query, expected_agent):
        with patch("src.workflow.router.get_llm", return_value=_MockRouterLLM()):
            from src.workflow.router import router_node
            state = _make_state(query)
            result = router_node(state)

        assert result["next_agent"] == expected_agent, (
            f"Query '{query}' routed to {result['next_agent'].value}, "
            f"expected {expected_agent.value}. "
            f"Reasoning: {result.get('router_reasoning', 'N/A')}"
        )

    def test_empty_message_defaults_to_finance_qa(self):
        """Router should default to finance_qa when no user message exists."""
        from src.workflow.router import router_node
        state = FinnieState(
            messages=[],
            user_profile=UserProfile(),
            financial_data=FinancialData(),
        )
        result = router_node(state)
        assert result["next_agent"] == AgentType.FINANCE_QA

    def test_router_handles_malformed_llm_response(self):
        """Router should fallback gracefully if LLM returns garbage."""
        bad_llm = MagicMock()
        bad_llm.invoke.return_value = MagicMock(content="This is not JSON at all!")

        with patch("src.workflow.router.get_llm", return_value=bad_llm):
            from src.workflow.router import router_node
            state = _make_state("Some question")
            result = router_node(state)

        # Should fallback to finance_qa
        assert result["next_agent"] == AgentType.FINANCE_QA

    def test_router_handles_llm_exception(self):
        """Router should fallback if the LLM throws an exception."""
        bad_llm = MagicMock()
        bad_llm.invoke.side_effect = RuntimeError("API timeout")

        with patch("src.workflow.router.get_llm", return_value=bad_llm):
            from src.workflow.router import router_node
            state = _make_state("Any question")
            result = router_node(state)

        assert result["next_agent"] == AgentType.FINANCE_QA
        assert "error" in result["router_reasoning"].lower()


# ---------------------------------------------------------------------------
# Live router tests (real Claude API)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not LIVE_MODE, reason="Set FINNIE_EVAL_LIVE=1 for live router evals")
class TestLiveRouterAccuracy:
    """Test router with real Claude API — the gold standard for routing accuracy."""

    @pytest.mark.parametrize(
        "query,expected_agent",
        ROUTING_CASES,
        ids=[f"{a.value}:{q[:40]}" for q, a in ROUTING_CASES],
    )
    def test_live_routing(self, query, expected_agent):
        from src.workflow.router import router_node
        state = _make_state(query)
        result = router_node(state)

        assert result["next_agent"] == expected_agent, (
            f"LIVE: Query '{query}' routed to {result['next_agent'].value}, "
            f"expected {expected_agent.value}. "
            f"Reasoning: {result.get('router_reasoning', 'N/A')}"
        )
