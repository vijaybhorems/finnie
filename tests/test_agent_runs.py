"""Run every agent's run() end to end with realistic provider data.

Unit tests elsewhere exercise helpers (ticker extraction, projections); this
file exercises the whole run() body. A refactor that leaves a dangling local
(the phase-1 `headlines` NameError shipped exactly that way) fails here
instead of on the first real news question.

The LLM is a real LangChain chat model (a recording fake), not a MagicMock, so
prompt assembly runs for real and the exact messages sent are inspectable.
"""
from __future__ import annotations

import importlib
import itertools
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from src.core.state import FinancialData, FinnieState, UserProfile


class RecordingChatModel(GenericFakeChatModel):
    """Fake chat model that records every message list it is invoked with."""

    calls: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _fake_llm() -> RecordingChatModel:
    return RecordingChatModel(messages=itertools.cycle([AIMessage(content="Educational answer.")]))


# Provider payloads shaped like the real clients' return values. Braces in the
# RAG text are deliberate: knowledge-base articles can contain JSON snippets.
_RAG = '[Source 1: Stocks And Bonds (investing_basics/stocks_and_bonds.txt)]\nExample: {"pe": 21.4}'
_PRICE = {"ticker": "SPY", "current_price": 512.3, "previous_close": 509.1}
_SECTORS = {"one_day": {"Technology": 1.2, "Energy": -0.4}}
_HEADLINE = {"title": "Fed holds rates", "source": "Reuters", "description": "Policy unchanged."}


def _patch_finance_qa(stack: ExitStack) -> None:
    retriever = stack.enter_context(patch("src.agents.finance_qa_agent.get_retriever"))
    retriever.return_value.get_context.return_value = _RAG
    fred = stack.enter_context(patch("src.agents.finance_qa_agent.FredClient"))
    fred.return_value.get_macro_snapshot.return_value = {
        "fed_funds_rate": {"value": 4.33, "date": "2026-09-01"},
    }


def _patch_tax(stack: ExitStack) -> None:
    retriever = stack.enter_context(patch("src.agents.tax_education_agent.get_retriever"))
    retriever.return_value.get_context.return_value = _RAG


def _patch_portfolio(stack: ExitStack) -> None:
    yf = stack.enter_context(patch("src.agents.portfolio_agent.YFinanceClient"))
    yf.return_value.get_portfolio_metrics.return_value = {
        "holdings": [{"ticker": "AAPL", "shares": 10, "current_value": 2300.0}],
        "total_value": 2300.0,
        "total_gain_loss": 800.0,
        "total_gain_loss_pct": 53.3,
    }
    av = stack.enter_context(patch("src.agents.portfolio_agent.AlphaVantageClient"))
    av.return_value.get_sector_performance.return_value = _SECTORS


def _patch_market(stack: ExitStack) -> None:
    yf = stack.enter_context(patch("src.agents.market_analysis_agent.YFinanceClient"))
    yf.return_value.get_current_price.return_value = _PRICE
    yf.return_value.get_sector_performance.return_value = _SECTORS
    av = stack.enter_context(patch("src.agents.market_analysis_agent.AlphaVantageClient"))
    av.return_value.get_rsi.return_value = {"rsi": 58.2, "signal": "neutral"}


def _patch_news(stack: ExitStack) -> None:
    news = stack.enter_context(patch("src.agents.news_synthesizer_agent.NewsClient"))
    news.return_value.get_financial_headlines.return_value = [_HEADLINE]
    news.return_value.get_sec_filings.return_value = [{"title": "8-K: AAPL results"}]
    news.return_value.get_ticker_news.return_value = [_HEADLINE]
    stack.enter_context(patch("src.agents.news_synthesizer_agent.YFinanceClient"))


def _patch_goal(stack: ExitStack) -> None:
    fred = stack.enter_context(patch("src.agents.goal_planning_agent.FredClient"))
    fred.return_value.get_interest_rate_environment.return_value = {"fed_funds_rate": 4.33}


_AGENTS = [
    ("finance_qa", "src.agents.finance_qa_agent", "FinanceQAAgent", _patch_finance_qa,
     "What is a P/E ratio?"),
    ("tax_education", "src.agents.tax_education_agent", "TaxEducationAgent", _patch_tax,
     "How does a Roth IRA work?"),
    ("portfolio", "src.agents.portfolio_agent", "PortfolioAgent", _patch_portfolio,
     "Analyze my portfolio: AAPL 10 shares @ $150"),
    ("market_analysis", "src.agents.market_analysis_agent", "MarketAnalysisAgent", _patch_market,
     "What is AAPL doing today?"),
    ("news_synthesizer", "src.agents.news_synthesizer_agent", "NewsSynthesizerAgent", _patch_news,
     "What happened in the market today?"),
    ("goal_planning", "src.agents.goal_planning_agent", "GoalPlanningAgent", _patch_goal,
     "How much do I need to retire at 55?"),
]
_IDS = [a[0] for a in _AGENTS]


def _state(query: str) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(portfolio=[{"ticker": "AAPL", "shares": 10, "avg_cost": 150.0}]),
        financial_data=FinancialData(),
    )


def run_agent(module: str, cls: str, patcher, query: str) -> tuple[dict[str, Any], RecordingChatModel]:
    """Construct the agent with patched providers and a recording LLM, run one turn."""
    llm = _fake_llm()
    with ExitStack() as stack:
        stack.enter_context(patch("src.agents.base_agent.get_llm", return_value=llm))
        patcher(stack)
        agent_cls = getattr(importlib.import_module(module), cls)
        result = agent_cls().run(_state(query))
    return result, llm


@pytest.mark.parametrize("name,module,cls,patcher,query", _AGENTS, ids=_IDS)
class TestEveryAgentRuns:
    def test_run_completes_with_disclaimer(self, name, module, cls, patcher, query):
        result, _ = run_agent(module, cls, patcher, query)

        assert result["final_response"].startswith("Educational answer.")
        assert "Disclaimer" in result["final_response"]
        assert result["messages"][0].content == result["final_response"]

    def test_llm_receives_system_prompt_and_question(self, name, module, cls, patcher, query):
        _, llm = run_agent(module, cls, patcher, query)

        assert len(llm.calls) == 1
        sent = llm.calls[0]
        assert sent[0].type == "system"
        assert sent[-1].type == "human" and sent[-1].content == query
