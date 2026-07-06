"""Financial-data population evals.

Verifies that data-driven agents (market, portfolio, news) populate the
structured ``FinancialData`` payload attached to their responses. Runs offline
with an echoing LLM stub and mocked data clients.

Run with:
    pytest tests/evals/test_financial_data_evals.py -v
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.core.state import FinancialData, FinnieState, UserProfile


def _echo_invoke_llm(self, state, additional_system=""):
    return "Educational analysis."


@contextmanager
def _patched_llm():
    with patch("src.agents.base_agent.get_llm"), \
         patch.object(
             __import__("src.agents.base_agent", fromlist=["BaseAgent"]).BaseAgent,
             "_invoke_llm",
             _echo_invoke_llm,
         ):
        yield


def _make_state(query: str, **profile_kwargs) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(**profile_kwargs),
        financial_data=FinancialData(),
    )


class TestMarketFinancialData:
    """Market agent populates tickers + price_data including requested symbols."""

    def test_populates_tickers_and_prices(self):
        with _patched_llm(), \
             patch("src.agents.market_analysis_agent.YFinanceClient") as mock_yf, \
             patch("src.agents.market_analysis_agent.AlphaVantageClient") as mock_av:
            mock_yf.return_value.get_current_price.return_value = {"current_price": 190.0}
            mock_av.return_value.get_sector_performance.return_value = {"one_day": {}}
            mock_av.return_value.get_rsi.return_value = {"rsi": 55}
            from src.agents.market_analysis_agent import MarketAnalysisAgent
            result = MarketAnalysisAgent().run(_make_state("What is AAPL doing today?"))

        fd = result["financial_data"]
        assert "AAPL" in fd.tickers
        assert "SPY" in fd.tickers  # index snapshot always included
        assert fd.price_data, "price_data should be populated"


class TestPortfolioFinancialData:
    """Portfolio agent populates tickers + summary metrics."""

    def test_populates_metrics(self):
        with _patched_llm(), \
             patch("src.agents.portfolio_agent.YFinanceClient") as mock_yf, \
             patch("src.agents.portfolio_agent.AlphaVantageClient") as mock_av:
            mock_yf.return_value.get_portfolio_metrics.return_value = {
                "holdings": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
                "total_value": 4500.0,
                "total_gain_loss": 300.0,
                "total_gain_loss_pct": 7.14,
            }
            mock_av.return_value.get_sector_performance.return_value = {"one_day": {}}
            from src.agents.portfolio_agent import PortfolioAgent
            result = PortfolioAgent().run(
                _make_state("Analyze my portfolio: AAPL 10 @ $150, MSFT 5 @ $300")
            )

        fd = result["financial_data"]
        assert fd.tickers == ["AAPL", "MSFT"]
        assert fd.metrics.get("total_value") == 4500.0


class TestNewsFinancialData:
    """News agent populates headlines + sources."""

    def test_populates_headlines(self):
        with _patched_llm(), \
             patch("src.agents.news_synthesizer_agent.NewsClient") as mock_news, \
             patch("src.agents.news_synthesizer_agent.YFinanceClient"):
            mock_news.return_value.get_financial_headlines.return_value = [
                {"title": "Fed holds rates", "source": "Reuters", "description": "d"},
                {"title": "Tech earnings beat", "source": "Bloomberg", "description": "d"},
            ]
            mock_news.return_value.get_sec_filings.return_value = []
            mock_news.return_value.get_ticker_news.return_value = []
            from src.agents.news_synthesizer_agent import NewsSynthesizerAgent
            result = NewsSynthesizerAgent().run(_make_state("What's the latest news?"))

        fd = result["financial_data"]
        assert "Fed holds rates" in fd.news_headlines
        assert set(fd.sources) == {"Reuters", "Bloomberg"}
