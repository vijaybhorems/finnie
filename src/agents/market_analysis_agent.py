"""Market Analysis Agent — real-time market insights and technical analysis."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.core.state import FinnieState
from src.utils.parallel import gather
from src.data.alpha_vantage_client import AlphaVantageClient
from src.data.yfinance_client import YFinanceClient


class MarketAnalysisAgent(BaseAgent):
    """Agent that analyzes market data, trends, and provides investment insights."""

    name = "Market Analysis Agent"
    description = (
        "I provide real-time market analysis including stock quotes, technical indicators "
        "(RSI, MACD), sector performance, and market trend education."
    )

    # Major indices to always include in snapshot
    INDEX_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]

    def __init__(self) -> None:
        super().__init__()
        self._yf = YFinanceClient()
        self._av = AlphaVantageClient()

    def run(self, state: FinnieState) -> dict[str, Any]:
        self._logger.info("market_analysis_agent_running")

        user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
        query = str(user_messages[-1].content) if user_messages else ""

        # Extract any tickers mentioned in the query
        mentioned_tickers = self._extract_tickers(query)

        # Every quote, the sector sweep and each ticker's indicators are
        # independent provider calls. Issued serially they dominated the turn;
        # run them all concurrently instead.
        focus_tickers = mentioned_tickers[:3]  # limit to avoid rate limits
        tasks: dict[str, Any] = {
            f"index:{ticker}": (lambda t=ticker: self._yf.get_current_price(t))
            for ticker in self.INDEX_TICKERS
        }
        tasks["sector"] = self._yf.get_sector_performance
        for ticker in focus_tickers:
            tasks[f"rsi:{ticker}"] = lambda t=ticker: self._av.get_rsi(t)
            tasks[f"price:{ticker}"] = lambda t=ticker: self._yf.get_current_price(t)

        fetched = gather(tasks)

        index_data = {t: fetched.get(f"index:{t}") for t in self.INDEX_TICKERS}
        sector_data = fetched.get("sector") or {}
        technical_data = {
            ticker: {
                "price": fetched.get(f"price:{ticker}"),
                "rsi": fetched.get(f"rsi:{ticker}"),
            }
            for ticker in focus_tickers
        }

        index_json = json.dumps(index_data, indent=2, default=str)
        sector_json = json.dumps(sector_data.get("one_day", {}), indent=2) if sector_data and "error" not in sector_data else "{}"
        tech_json = json.dumps(technical_data, indent=2, default=str) if technical_data else "No specific tickers requested."

        additional_system = f"""
{self._get_user_context_str(state)}

MARKET INDEX SNAPSHOT (current):
{index_json}

SECTOR PERFORMANCE (1-day):
{sector_json}

TECHNICAL ANALYSIS FOR REQUESTED TICKERS:
{tech_json}

Provide market analysis covering:
1. Current market conditions summary (based on index data)
2. Notable sector movements (if any)
3. Technical analysis for any requested tickers (explain RSI, trends in plain language)
4. Educational context: what does today's market environment mean for different investor types?

Explain technical indicators (RSI, moving averages) educationally for beginners.
"""

        response_text = self._invoke_llm(state, additional_system)
        response_text = self._add_disclaimer(response_text)

        updated_financial_data = state.financial_data.model_copy()
        updated_financial_data.tickers = list(index_data.keys()) + mentioned_tickers
        updated_financial_data.price_data = {**index_data, **{t: technical_data[t]["price"] for t in technical_data}}

        return {
            "messages": [AIMessage(content=response_text, name=self.name)],
            "final_response": response_text,
            "financial_data": updated_financial_data,
        }

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract stock ticker symbols from text."""
        # Common false positives to exclude
        exclude = {"I", "A", "AT", "IT", "BE", "BY", "OF", "OR", "IS", "TO", "DO", "GO", "NO"}
        # Look for 1-5 uppercase letters that appear to be tickers
        candidates = re.findall(r"\b([A-Z]{1,5})\b", text)
        return [c for c in candidates if c not in exclude][:5]
