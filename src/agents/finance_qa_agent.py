"""Finance Q&A Agent — general financial education using RAG + FRED data."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.core.state import FinnieState
from src.data.companies import company_headlines, format_companies, research_companies
from src.data.fred_client import FredClient
from src.data.yfinance_client import YFinanceClient
from src.rag.retriever import get_retriever
from src.utils.parallel import gather


class FinanceQAAgent(BaseAgent):
    """Agent that answers general financial questions using RAG retrieval and live FRED macro data."""

    name = "Finance Q&A Agent"
    description = (
        "I specialize in financial education — explaining concepts like investing, "
        "stocks, bonds, ETFs, diversification, and macroeconomic principles in clear, "
        "accessible language for all knowledge levels."
    )
    prompt_name = "finance_qa"

    def __init__(self) -> None:
        super().__init__()
        self._retriever = get_retriever()
        self._fred = FredClient()
        self._yf = YFinanceClient()

    def run(self, state: FinnieState) -> dict[str, Any]:
        self._logger.info("finance_qa_agent_running")

        # Extract the latest user query
        user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
        query = str(user_messages[-1].content) if user_messages else ""

        # RAG and the macro snapshot are independent network/disk calls, so
        # fetch them concurrently. Macro is skipped entirely when the classifier
        # judged it irrelevant (needs_macro is None on the legacy path, which
        # always fetched — preserve that).
        want_macro = state.needs_macro is not False
        tasks = {"rag": lambda: self._retriever.get_context(query, top_k=4)}
        if want_macro:
            tasks["macro"] = self._fred.get_macro_snapshot
        # A question naming a company ("Should I care about the Anthropic IPO?")
        # gets its live price and headlines, so the answer isn't limited to
        # what the model remembers.
        named = state.query_companies or state.query_tickers
        if named:
            tasks["companies"] = lambda: research_companies(
                state.query_companies, state.query_tickers, self._yf, news_count=3,
            )
        fetched = gather(tasks)
        companies = fetched.get("companies") or []

        rag_context = fetched.get("rag") or ""
        macro_str = (
            self._format_macro(fetched.get("macro") or {})
            if want_macro
            else "Not fetched — this question does not depend on current macro readings."
        )

        context = f"""
{self._get_user_context_str(state)}

KNOWLEDGE BASE CONTEXT (use this to ground your answer):
{rag_context if rag_context else "No specific knowledge base articles found for this query."}

CURRENT MACROECONOMIC DATA (reference if relevant):
{macro_str}
"""
        if named:
            context += f"""
COMPANIES NAMED IN THE QUESTION (live search: price and latest news):
{format_companies(companies) if companies else "Live company data could not be retrieved right now."}
"""

        response_text = self._invoke_llm(state, context)
        response_text = self._add_disclaimer(response_text)

        update: dict[str, Any] = {
            "messages": [AIMessage(content=response_text, name=self.name)],
            "final_response": response_text,
            "rag_context": rag_context.split("\n\n---\n\n") if rag_context else [],
        }
        if companies:
            # Live data dates the answer: recording it keeps the FAQ cache from
            # storing it (faq_cache._used_live_data).
            data = state.financial_data.model_copy()
            data.tickers = [c["symbol"] for c in companies if c["symbol"]]
            data.price_data = {c["symbol"]: c["price"] for c in companies if c["symbol"] and c["price"]}
            data.news_headlines = company_headlines(companies)
            update["financial_data"] = data
        return update

    def _format_macro(self, macro: dict) -> str:
        lines = []
        mappings = {
            "fed_funds_rate": "Fed Funds Rate",
            "cpi": "CPI (Consumer Price Index)",
            "gdp_growth": "GDP Growth Rate",
            "unemployment": "Unemployment Rate",
            "10yr_treasury": "10-Year Treasury Yield",
            "inflation_expectations": "5-Year Inflation Expectations",
        }
        for key, label in mappings.items():
            if key in macro and "error" not in macro[key]:
                val = macro[key]
                lines.append(f"- {label}: {val['value']} (as of {val['date']})")
        return "\n".join(lines) if lines else "Macro data unavailable"
