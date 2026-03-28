"""Tests for individual agents."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.core.state import FinancialData, FinnieState, UserProfile


def _make_state(query: str, profile: dict | None = None) -> FinnieState:
    profile_obj = UserProfile(**(profile or {}))
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=profile_obj,
        financial_data=FinancialData(),
    )


class TestFinanceQAAgent:
    @patch("src.agents.finance_qa_agent.get_retriever")
    @patch("src.agents.base_agent.get_llm")
    def test_run_returns_response(self, mock_llm, mock_retriever):
        mock_llm.return_value.invoke.return_value = MagicMock(content="A P/E ratio is...")

        mock_ret = MagicMock()
        mock_ret.get_context.return_value = "P/E ratio context from knowledge base"
        mock_retriever.return_value = mock_ret

        with patch("src.agents.finance_qa_agent.FredClient") as mock_fred:
            mock_fred.return_value.get_macro_snapshot.return_value = {}

            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
            state = _make_state("What is a P/E ratio?")
            result = agent.run(state)

        assert "final_response" in result
        assert len(result["final_response"]) > 0
        assert "messages" in result

    @patch("src.agents.finance_qa_agent.get_retriever")
    @patch("src.agents.base_agent.get_llm")
    def test_response_includes_disclaimer(self, mock_llm, mock_retriever):
        mock_llm.return_value.invoke.return_value = MagicMock(content="Educational content")
        mock_retriever.return_value = MagicMock(get_context=MagicMock(return_value=""))

        with patch("src.agents.finance_qa_agent.FredClient") as mock_fred:
            mock_fred.return_value.get_macro_snapshot.return_value = {}

            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
            state = _make_state("Test question")
            result = agent.run(state)

        assert "educational" in result["final_response"].lower() or "disclaimer" in result["final_response"].lower() or "not financial advice" in result["final_response"].lower()


class TestPortfolioAgent:
    @patch("src.agents.portfolio_agent.AlphaVantageClient")
    @patch("src.agents.portfolio_agent.YFinanceClient")
    @patch("src.agents.base_agent.get_llm")
    def test_parses_portfolio_from_message(self, mock_llm, mock_yf, mock_av):
        mock_llm.return_value.invoke.return_value = MagicMock(content="Portfolio analysis...")
        mock_yf.return_value.get_portfolio_metrics.return_value = {
            "holdings": [],
            "total_value": 0,
            "total_cost": 0,
            "total_gain_loss": 0,
            "total_gain_loss_pct": 0,
        }
        mock_av.return_value.get_sector_performance.return_value = {}

        from src.agents.portfolio_agent import PortfolioAgent
        agent = PortfolioAgent()
        state = _make_state("AAPL: 10 shares @ $150, MSFT: 5 shares @ $300")
        holdings = agent._parse_portfolio_from_message(state)

        assert len(holdings) >= 1
        tickers = [h["ticker"] for h in holdings]
        assert "AAPL" in tickers

    def test_parse_portfolio_no_holdings(self):
        from src.agents.portfolio_agent import PortfolioAgent
        with patch("src.agents.portfolio_agent.YFinanceClient"), \
             patch("src.agents.portfolio_agent.AlphaVantageClient"), \
             patch("src.agents.base_agent.get_llm"):
            agent = PortfolioAgent()
        state = _make_state("What is diversification?")
        holdings = agent._parse_portfolio_from_message(state)
        assert holdings == []


class TestGoalPlanningAgent:
    def test_project_savings_basic(self):
        from src.agents.goal_planning_agent import _project_savings
        # 0% return: should just be contributions + initial
        result = _project_savings(500, 0, 0.0, 10)
        assert result == pytest.approx(500 * 12 * 10, rel=0.01)

    def test_project_savings_compound(self):
        from src.agents.goal_planning_agent import _project_savings
        # $10k at 7% for 10 years should approximately double
        result = _project_savings(0, 10000, 0.07, 10)
        assert result > 19000  # roughly 2x

    def test_years_to_goal_reachable(self):
        from src.agents.goal_planning_agent import _years_to_goal
        years = _years_to_goal(100000, 500, 10000, 0.07)
        assert 0 < years < 40

    def test_years_to_goal_already_reached(self):
        from src.agents.goal_planning_agent import _years_to_goal
        years = _years_to_goal(5000, 500, 10000, 0.07)
        assert years <= 1


class TestTaxEducationAgent:
    @patch("src.agents.tax_education_agent.get_retriever")
    @patch("src.agents.base_agent.get_llm")
    def test_run_returns_response(self, mock_llm, mock_retriever):
        mock_llm.return_value.invoke.return_value = MagicMock(content="Tax education content...")
        mock_ret = MagicMock()
        mock_ret.get_context.return_value = "Tax knowledge base context"
        mock_retriever.return_value = mock_ret

        from src.agents.tax_education_agent import TaxEducationAgent
        agent = TaxEducationAgent()
        state = _make_state("What are capital gains taxes?")
        result = agent.run(state)

        assert "final_response" in result
        assert len(result["final_response"]) > 0


class TestMarketAnalysisAgent:
    def test_extract_tickers(self):
        from src.agents.market_analysis_agent import MarketAnalysisAgent
        with patch("src.agents.market_analysis_agent.YFinanceClient"), \
             patch("src.agents.market_analysis_agent.AlphaVantageClient"), \
             patch("src.agents.base_agent.get_llm"):
            agent = MarketAnalysisAgent()

        tickers = agent._extract_tickers("What is AAPL doing? Also check MSFT and TSLA")
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "TSLA" in tickers

    def test_extract_tickers_filters_common_words(self):
        from src.agents.market_analysis_agent import MarketAnalysisAgent
        with patch("src.agents.market_analysis_agent.YFinanceClient"), \
             patch("src.agents.market_analysis_agent.AlphaVantageClient"), \
             patch("src.agents.base_agent.get_llm"):
            agent = MarketAnalysisAgent()

        tickers = agent._extract_tickers("I want to invest in the market")
        # "I" and "A" should be filtered out
        assert "I" not in tickers
        assert "A" not in tickers
