"""Companies named in a question, looked up live: Yahoo search, the lookup, and the agents.

A company that listed after the model's training has no symbol the model knows
("What's going on with the Anthropic IPO?" a day after it listed). The lookup
resolves the name and brings its recent news in the same call.
"""
from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.core.state import FinancialData, FinnieState, UserProfile
from src.data.companies import format_companies, research_companies
from tests.test_agent_runs import _fake_llm

_NEWS = {"title": "Example Robotics jumps in market debut", "source": "Wire",
         "url": "https://example.com/a", "published_at": "2026-11-21T14:00+00:00"}
_QUOTE = {"symbol": "EXRB", "name": "Example Robotics, Inc.", "exchange": "NASDAQ", "type": "EQUITY"}


class FakeYF:
    """search() and get_current_price() shaped like YFinanceClient's."""

    def __init__(self, results: dict[str, dict] | None = None):
        self.results = results or {}
        self.searched: list[str] = []
        self.priced: list[str] = []

    def search(self, query, news_count=5):
        self.searched.append(query)
        return self.results.get(query, {"quotes": [], "news": []})

    def get_current_price(self, ticker):
        self.priced.append(ticker)
        return {"ticker": ticker, "current_price": 42.5, "previous_close": 40.0,
                "change_pct": 6.25, "currency": "USD"}


class TestYahooSearch:
    @patch("src.data.yfinance_client.yf.Search")
    @patch("src.data.yfinance_client.get_cache")
    def test_keeps_us_listed_securities_and_maps_news(self, mock_cache, mock_search):
        from src.data.yfinance_client import YFinanceClient

        mock_cache.return_value.get.return_value = None
        mock_search.return_value = SimpleNamespace(
            quotes=[
                {"symbol": "EXRB", "longname": "Example Robotics, Inc.", "exchDisp": "NASDAQ", "quoteType": "EQUITY"},
                {"symbol": "SEXRB=F", "quoteType": "FUTURE"},
                {"symbol": "EXRB.VI", "longname": "Example Robotics", "quoteType": "EQUITY"},
                {"symbol": "EXRU", "shortname": "2x Example Robotics ETF", "exchange": "NGM", "quoteType": "ETF"},
            ],
            news=[{"title": "Debut", "publisher": "Wire", "link": "https://x", "providerPublishTime": 1795489621},
                  {"publisher": "no title"}],
        )
        result = YFinanceClient().search("  Example   Robotics ")

        assert mock_search.call_args.args[0] == "Example Robotics"
        assert [q["symbol"] for q in result["quotes"]] == ["EXRB", "EXRU"]
        assert result["quotes"][1]["name"] == "2x Example Robotics ETF"
        assert result["news"] == [{"title": "Debut", "source": "Wire", "url": "https://x",
                                   "published_at": "2026-11-24T03:07+00:00"}]

    @patch("src.data.yfinance_client.yf.Search", side_effect=RuntimeError("yahoo down"))
    @patch("src.data.yfinance_client.get_cache")
    def test_failure_returns_empty(self, mock_cache, _search):
        from src.data.yfinance_client import YFinanceClient

        mock_cache.return_value.get.return_value = None
        mock_cache.return_value.get_stale.return_value = None
        assert YFinanceClient().search("Example Robotics") == {"quotes": [], "news": []}


class TestResearchCompanies:
    def test_resolves_a_name_the_model_has_no_symbol_for(self):
        yf = FakeYF({"Example Robotics": {"quotes": [_QUOTE], "news": [_NEWS]}})
        [company] = research_companies(["Example Robotics"], [], yf)

        assert company["symbol"] == "EXRB"
        assert company["name"] == "Example Robotics, Inc."
        assert company["news"] == [_NEWS]
        assert yf.priced == ["EXRB"]

    def test_a_ticker_the_name_already_resolved_is_not_repeated(self):
        yf = FakeYF({"Example Robotics": {"quotes": [_QUOTE], "news": [_NEWS]}})
        companies = research_companies(["Example Robotics"], ["EXRB"], yf)

        assert [c["symbol"] for c in companies] == ["EXRB"]

    def test_unknown_name_has_no_symbol_and_is_not_priced(self):
        yf = FakeYF()
        [company] = research_companies(["Nonexistent Co"], [], yf)

        assert company["symbol"] is None
        assert yf.priced == []
        assert "no listed security found" in format_companies([company])

    def test_a_ticker_search_doesnt_match_is_kept(self):
        yf = FakeYF({"AAPL": {"quotes": [{**_QUOTE, "symbol": "AAPU"}], "news": []}})
        [company] = research_companies([], ["AAPL"], yf)

        assert company["symbol"] == "AAPL" and company["name"] == ""

    def test_nothing_named_makes_no_calls(self):
        yf = FakeYF()
        assert research_companies([], [], yf) == []
        assert yf.searched == []

    def test_format_shows_price_and_dated_news(self):
        yf = FakeYF({"Example Robotics": {"quotes": [_QUOTE], "news": [_NEWS]}})
        text = format_companies(research_companies(["Example Robotics"], [], yf))

        assert "Example Robotics, Inc. (EXRB, NASDAQ)" in text
        assert "42.50 USD (+6.25% vs previous close)" in text
        assert "[Wire] Example Robotics jumps in market debut (2026-11-21)" in text


def _state(query: str, companies: list[str], tickers: list[str] | None = None) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(),
        financial_data=FinancialData(),
        query_companies=companies,
        query_tickers=tickers or [],
    )


def _context(llm) -> str:
    """The per-request context block the agent sent."""
    system = llm.calls[0][0]
    return "\n".join(b["text"] if isinstance(b, dict) else str(b) for b in system.content) \
        if isinstance(system.content, list) else str(system.content)


class TestAgentsUseNamedCompanies:
    def _patch_yf(self, stack, module):
        yf_cls = stack.enter_context(patch(f"{module}.YFinanceClient"))
        fake = FakeYF({"Example Robotics": {"quotes": [_QUOTE], "news": [_NEWS]}})
        yf_cls.return_value.search.side_effect = fake.search
        yf_cls.return_value.get_current_price.side_effect = fake.get_current_price
        yf_cls.return_value.get_sector_performance.return_value = {}
        return fake

    def test_news_agent_brings_the_company_news_and_searches_its_name(self):
        from src.agents.news_synthesizer_agent import NewsSynthesizerAgent

        llm = _fake_llm()
        with ExitStack() as stack:
            stack.enter_context(patch("src.agents.base_agent.get_llm", return_value=llm))
            news = stack.enter_context(patch("src.agents.news_synthesizer_agent.NewsClient"))
            news.return_value.get_financial_headlines.return_value = []
            news.return_value.get_sec_filings.return_value = []
            self._patch_yf(stack, "src.agents.news_synthesizer_agent")
            result = NewsSynthesizerAgent().run(
                _state("What's going on with Example Robotics since the IPO?", ["Example Robotics"]))

        assert news.return_value.get_financial_headlines.call_args.kwargs["query"] == '"Example Robotics"'
        assert "Example Robotics jumps in market debut" in _context(llm)
        assert result["financial_data"].tickers == ["EXRB"]
        assert "Example Robotics jumps in market debut" in result["financial_data"].news_headlines

    def test_market_agent_fetches_the_resolved_symbol(self):
        from src.agents.market_analysis_agent import MarketAnalysisAgent

        with ExitStack() as stack:
            stack.enter_context(patch("src.agents.base_agent.get_llm", return_value=_fake_llm()))
            fake = self._patch_yf(stack, "src.agents.market_analysis_agent")
            av = stack.enter_context(patch("src.agents.market_analysis_agent.AlphaVantageClient"))
            av.return_value.get_rsi.return_value = {"rsi": 50}
            MarketAnalysisAgent().run(_state("How is Example Robotics trading?", ["Example Robotics"]))

        assert "EXRB" in fake.priced
        assert av.return_value.get_rsi.call_args.args[0] == "EXRB"

    def test_finance_qa_adds_live_company_data_and_marks_it(self):
        from src.agents.finance_qa_agent import FinanceQAAgent

        llm = _fake_llm()
        with ExitStack() as stack:
            stack.enter_context(patch("src.agents.base_agent.get_llm", return_value=llm))
            stack.enter_context(patch("src.agents.finance_qa_agent.get_retriever")).return_value \
                .get_context.return_value = ""
            stack.enter_context(patch("src.agents.finance_qa_agent.FredClient"))
            self._patch_yf(stack, "src.agents.finance_qa_agent")
            state = _state("Should a beginner care about the Example Robotics IPO?", ["Example Robotics"])
            state.needs_macro = False
            result = FinanceQAAgent().run(state)

        assert "Example Robotics, Inc. (EXRB, NASDAQ)" in _context(llm)
        assert result["financial_data"].tickers == ["EXRB"]  # keeps it out of the FAQ cache

    def test_finance_qa_without_a_company_makes_no_lookup(self):
        from src.agents.finance_qa_agent import FinanceQAAgent

        with ExitStack() as stack:
            stack.enter_context(patch("src.agents.base_agent.get_llm", return_value=_fake_llm()))
            stack.enter_context(patch("src.agents.finance_qa_agent.get_retriever")).return_value \
                .get_context.return_value = ""
            stack.enter_context(patch("src.agents.finance_qa_agent.FredClient"))
            yf_cls = stack.enter_context(patch("src.agents.finance_qa_agent.YFinanceClient"))
            state = _state("What is a P/E ratio?", [])
            state.needs_macro = False
            result = FinanceQAAgent().run(state)

        yf_cls.return_value.search.assert_not_called()
        assert "financial_data" not in result
