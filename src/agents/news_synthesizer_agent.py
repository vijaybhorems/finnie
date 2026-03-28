"""News Synthesizer Agent — contextualizes financial news for investors."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.core.state import FinnieState
from src.data.news_client import NewsClient
from src.data.yfinance_client import YFinanceClient


class NewsSynthesizerAgent(BaseAgent):
    """Agent that fetches and synthesizes financial news into concise, actionable summaries."""

    name = "News Synthesizer Agent"
    description = (
        "I synthesize and contextualize financial news, helping you understand how "
        "headlines, earnings reports, and economic events might affect markets and portfolios — "
        "from an educational perspective."
    )

    def __init__(self) -> None:
        super().__init__()
        self._news = NewsClient()
        self._yf = YFinanceClient()

    def run(self, state: FinnieState) -> dict[str, Any]:
        self._logger.info("news_synthesizer_agent_running")

        user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
        query = str(user_messages[-1].content) if user_messages else "latest financial news"

        # Fetch news
        headlines = self._news.get_financial_headlines(query=query, page_size=8)
        sec_filings = self._news.get_sec_filings(max_items=5)

        headlines_text = self._format_headlines(headlines)
        sec_text = self._format_filings(sec_filings)

        # Check for portfolio tickers in news
        portfolio_tickers = [h["ticker"] for h in state.user_profile.portfolio]
        ticker_news_text = ""
        for ticker in portfolio_tickers[:3]:
            news = self._news.get_ticker_news(ticker, page_size=3)
            if news:
                ticker_news_text += f"\n\n{ticker} News:\n" + self._format_headlines(news)

        additional_system = f"""
{self._get_user_context_str(state)}

LATEST FINANCIAL HEADLINES:
{headlines_text}

RECENT SEC FILINGS (8-K Events):
{sec_text}

PORTFOLIO-SPECIFIC NEWS:
{ticker_news_text if ticker_news_text else "No portfolio holdings to track."}

Synthesize the news for this investor:
1. Identify the 2-3 most significant stories and why they matter
2. Explain market implications in plain language
3. Connect news to broader economic context (interest rates, inflation, growth)
4. If relevant to their portfolio, highlight potential impact (educational framing)
5. Help them distinguish signal from noise in financial media

Keep the tone educational — help them develop financial news literacy.
"""

        response_text = self._invoke_llm(state, additional_system)
        response_text = self._add_disclaimer(response_text)

        updated_financial_data = state.financial_data.model_copy()
        updated_financial_data.news_headlines = [h["title"] for h in headlines]
        updated_financial_data.sources = list({h["source"] for h in headlines})

        return {
            "messages": [AIMessage(content=response_text, name=self.name)],
            "final_response": response_text,
            "financial_data": updated_financial_data,
        }

    def _format_headlines(self, articles: list[dict]) -> str:
        if not articles:
            return "No recent headlines available."
        lines = []
        for a in articles:
            lines.append(f"• [{a.get('source', 'Unknown')}] {a.get('title', 'No title')}")
            if a.get("description"):
                lines.append(f"  {a['description'][:150]}...")
        return "\n".join(lines)

    def _format_filings(self, filings: list[dict]) -> str:
        if not filings:
            return "No recent SEC filings available."
        return "\n".join(f"• {f.get('title', '')[:100]}" for f in filings)
