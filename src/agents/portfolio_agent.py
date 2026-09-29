"""Portfolio Analysis Agent — analyzes user portfolios with metrics and recommendations."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.core.state import FinnieState
from src.data.alpha_vantage_client import AlphaVantageClient
from src.data.yfinance_client import YFinanceClient
from src.utils.parallel import gather


class PortfolioAgent(BaseAgent):
    """Agent that analyzes investment portfolios and calculates key metrics and diversification insights."""

    name = "Portfolio Analysis Agent"
    description = (
        "I specialize in analyzing investment portfolios. I calculate key metrics like "
        "P/E ratios, beta, sector diversification, gain/loss, and provide educational "
        "insights about portfolio health and diversification."
    )
    prompt_name = "portfolio"

    def __init__(self) -> None:
        super().__init__()
        self._yf = YFinanceClient()
        self._av = AlphaVantageClient()

    def run(self, state: FinnieState) -> dict[str, Any]:
        self._logger.info("portfolio_agent_running")

        portfolio = state.user_profile.portfolio
        if not portfolio:
            # Try to parse portfolio from latest message
            portfolio = self._parse_portfolio_from_message(state)

        # Holdings metrics and sector performance hit different providers —
        # fetch concurrently rather than one after the other.
        tasks: dict[str, Any] = {"sector": self._av.get_sector_performance}
        if portfolio:
            tasks["portfolio"] = lambda: self._yf.get_portfolio_metrics(portfolio)
        fetched = gather(tasks)

        portfolio_data: dict[str, Any] = fetched.get("portfolio") or {}
        if portfolio and not portfolio_data:
            portfolio_data = {"error": "portfolio metrics unavailable"}
        sector_performance = fetched.get("sector") or {}

        portfolio_json = json.dumps(portfolio_data, indent=2, default=str) if portfolio_data else "No portfolio data provided."
        sector_json = json.dumps(sector_performance.get("one_day", {}), indent=2) if sector_performance and "error" not in sector_performance else "{}"

        context = f"""
{self._get_user_context_str(state)}

PORTFOLIO DATA:
{portfolio_json}

MARKET SECTOR PERFORMANCE (1-day):
{sector_json}
"""

        response_text = self._invoke_llm(state, context)
        response_text = self._add_disclaimer(response_text)

        updated_financial_data = state.financial_data.model_copy()
        if portfolio_data and "holdings" in portfolio_data:
            updated_financial_data.tickers = [h["ticker"] for h in portfolio_data["holdings"]]
            updated_financial_data.metrics = {
                "total_value": portfolio_data.get("total_value"),
                "total_gain_loss": portfolio_data.get("total_gain_loss"),
                "total_gain_loss_pct": portfolio_data.get("total_gain_loss_pct"),
            }

        return {
            "messages": [AIMessage(content=response_text, name=self.name)],
            "final_response": response_text,
            "financial_data": updated_financial_data,
        }

    def _parse_portfolio_from_message(self, state: FinnieState) -> list[dict[str, Any]]:
        """Attempt to extract ticker/shares/cost from the latest user message."""
        user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
        if not user_messages:
            return []

        text = str(user_messages[-1].content)
        holdings = []
        import re

        # Pattern: "AAPL: 10 shares @ $150" or "10 AAPL at 150"
        patterns = [
            r"([A-Z]{1,5})[:\s]+(\d+(?:\.\d+)?)\s*(?:shares?)?\s*[@at]+\s*\$?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s+([A-Z]{1,5})\s+(?:at|@)\s+\$?(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) == 3:
                    ticker, shares, cost = groups
                    if not ticker.isalpha():
                        ticker, shares = shares, ticker
                    holdings.append({
                        "ticker": ticker.upper(),
                        "shares": float(shares),
                        "avg_cost": float(cost),
                    })

        return holdings
