"""Disclaimer coverage evals.

Every agent's response must carry the educational disclaimer ("not financial
advice"). This eval runs all six agents with an echoing LLM stub and mocked
data clients so it stays deterministic and offline.

Run with:
    pytest tests/evals/test_disclaimer_evals.py -v
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.core.state import FinancialData, FinnieState, UserProfile

_DISCLAIMER_MARKER = "not financial advice"


def _echo_invoke_llm(self, state, additional_system=""):
    return "Here is an educational explanation."


def _make_state(query: str, **profile_kwargs) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(**profile_kwargs),
        financial_data=FinancialData(),
    )


@contextmanager
def _patched_llm():
    """Patch BaseAgent LLM construction + invocation for offline runs."""
    with patch("src.agents.base_agent.get_llm"), \
         patch.object(
             __import__("src.agents.base_agent", fromlist=["BaseAgent"]).BaseAgent,
             "_invoke_llm",
             _echo_invoke_llm,
         ):
        yield


class TestDisclaimerCoverage:
    """All six agents append the educational disclaimer to every response."""

    def test_finance_qa(self, retriever):
        with _patched_llm(), \
             patch("src.agents.finance_qa_agent.get_retriever", return_value=retriever), \
             patch("src.agents.finance_qa_agent.FredClient") as mock_fred:
            mock_fred.return_value.get_macro_snapshot.return_value = {}
            from src.agents.finance_qa_agent import FinanceQAAgent
            result = FinanceQAAgent().run(_make_state("What is an ETF?"))
        assert _DISCLAIMER_MARKER in result["final_response"].lower()

    def test_tax_education(self, retriever):
        with _patched_llm(), \
             patch("src.agents.tax_education_agent.get_retriever", return_value=retriever):
            from src.agents.tax_education_agent import TaxEducationAgent
            result = TaxEducationAgent().run(_make_state("How does a Roth IRA work?"))
        assert _DISCLAIMER_MARKER in result["final_response"].lower()

    def test_goal_planning(self):
        with _patched_llm(), \
             patch("src.agents.goal_planning_agent.FredClient") as mock_fred:
            mock_fred.return_value.get_interest_rate_environment.return_value = {}
            from src.agents.goal_planning_agent import GoalPlanningAgent
            result = GoalPlanningAgent().run(_make_state("How much to retire at 60?"))
        assert _DISCLAIMER_MARKER in result["final_response"].lower()

    def test_market_analysis(self):
        with _patched_llm(), \
             patch("src.agents.market_analysis_agent.YFinanceClient") as mock_yf, \
             patch("src.agents.market_analysis_agent.AlphaVantageClient") as mock_av:
            mock_yf.return_value.get_current_price.return_value = {"current_price": 100}
            mock_av.return_value.get_sector_performance.return_value = {"one_day": {}}
            mock_av.return_value.get_rsi.return_value = {}
            from src.agents.market_analysis_agent import MarketAnalysisAgent
            result = MarketAnalysisAgent().run(_make_state("How is the market today?"))
        assert _DISCLAIMER_MARKER in result["final_response"].lower()

    def test_portfolio(self):
        with _patched_llm(), \
             patch("src.agents.portfolio_agent.YFinanceClient") as mock_yf, \
             patch("src.agents.portfolio_agent.AlphaVantageClient") as mock_av:
            mock_yf.return_value.get_portfolio_metrics.return_value = {
                "holdings": [{"ticker": "AAPL"}],
                "total_value": 1500.0,
                "total_gain_loss": 100.0,
                "total_gain_loss_pct": 7.1,
            }
            mock_av.return_value.get_sector_performance.return_value = {"one_day": {}}
            from src.agents.portfolio_agent import PortfolioAgent
            result = PortfolioAgent().run(_make_state("Analyze AAPL: 10 shares @ $150"))
        assert _DISCLAIMER_MARKER in result["final_response"].lower()

    def test_news_synthesizer(self):
        with _patched_llm(), \
             patch("src.agents.news_synthesizer_agent.NewsClient") as mock_news, \
             patch("src.agents.news_synthesizer_agent.YFinanceClient"):
            mock_news.return_value.get_financial_headlines.return_value = [
                {"title": "Markets rally", "source": "Reuters", "description": "d"}
            ]
            mock_news.return_value.get_sec_filings.return_value = []
            mock_news.return_value.get_ticker_news.return_value = []
            from src.agents.news_synthesizer_agent import NewsSynthesizerAgent
            result = NewsSynthesizerAgent().run(_make_state("Any market news?"))
        assert _DISCLAIMER_MARKER in result["final_response"].lower()
