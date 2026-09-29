"""Finance Q&A Agent — general financial education using RAG + FRED data."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.core.state import FinnieState
from src.data.fred_client import FredClient
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

    def __init__(self) -> None:
        super().__init__()
        self._retriever = get_retriever()
        self._fred = FredClient()

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
        fetched = gather(tasks)

        rag_context = fetched.get("rag") or ""
        macro_str = (
            self._format_macro(fetched.get("macro") or {})
            if want_macro
            else "Not fetched — this question does not depend on current macro readings."
        )

        additional_system = f"""
{self._get_user_context_str(state)}

KNOWLEDGE BASE CONTEXT (use this to ground your answer):
{rag_context if rag_context else "No specific knowledge base articles found for this query."}

CURRENT MACROECONOMIC DATA (reference if relevant):
{macro_str}

Answer the user's financial question comprehensively but concisely.
Adapt your explanation depth to their knowledge level.
Always cite sources (knowledge base articles or FRED data) when relevant.
"""

        response_text = self._invoke_llm(state, additional_system)
        response_text = self._add_disclaimer(response_text)

        return {
            "messages": [AIMessage(content=response_text, name=self.name)],
            "final_response": response_text,
            "rag_context": rag_context.split("\n\n---\n\n") if rag_context else [],
        }

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
