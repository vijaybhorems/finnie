"""Alpha Vantage client — real-time quotes, technical indicators, fundamentals."""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

from src.core.config import get_settings
from src.utils.cache import get_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RATE_LIMIT_DELAY = 12  # seconds between calls on free tier (5/min)


class AlphaVantageClient:
    """Alpha Vantage REST wrapper with rate-limit protection and caching."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cache = get_cache()
        self._api_key = self._settings.alpha_vantage_api_key
        self._base_url = self._settings.apis.alpha_vantage_base_url
        self._last_call: float = 0.0

    def _call(self, params: dict[str, str]) -> dict[str, Any]:
        if not self._api_key:
            return {"error": "Alpha Vantage API key not configured"}

        # Rate limit enforcement (free tier: 5 calls/min)
        elapsed = time.time() - self._last_call
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)

        params["apikey"] = self._api_key
        try:
            resp = requests.get(self._base_url, params=params, timeout=10)
            resp.raise_for_status()
            self._last_call = time.time()
            data = resp.json()
            if "Note" in data:
                logger.warning("alpha_vantage_rate_limit_note", note=data["Note"])
            return data
        except Exception as exc:
            logger.error("alpha_vantage_request_error", error=str(exc), params=params)
            return {"error": str(exc)}

    def get_quote(self, ticker: str) -> dict[str, Any]:
        """Global quote — latest price, volume, change%."""
        key = self._cache.cache_key("av", "quote", ticker.upper())
        cached = self._cache.get(key)
        if cached:
            return cached

        data = self._call({"function": "GLOBAL_QUOTE", "symbol": ticker})
        quote = data.get("Global Quote", {})
        result = {
            "ticker": ticker.upper(),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_pct": quote.get("10. change percent"),
            "volume": quote.get("06. volume"),
            "latest_trading_day": quote.get("07. latest trading day"),
            "previous_close": quote.get("08. previous close"),
            "open": quote.get("02. open"),
            "high": quote.get("03. high"),
            "low": quote.get("04. low"),
        }
        if "error" not in data:
            self._cache.set(key, result, ttl=300)
        return result

    def get_rsi(self, ticker: str, interval: str = "daily", time_period: int = 14) -> dict[str, Any]:
        """RSI technical indicator."""
        key = self._cache.cache_key("av", "rsi", ticker.upper(), interval, str(time_period))
        cached = self._cache.get(key)
        if cached:
            return cached

        data = self._call({
            "function": "RSI",
            "symbol": ticker,
            "interval": interval,
            "time_period": str(time_period),
            "series_type": "close",
        })
        technical = data.get("Technical Analysis: RSI", {})
        if technical:
            dates = sorted(technical.keys(), reverse=True)[:5]
            result = {
                "ticker": ticker.upper(),
                "indicator": "RSI",
                "period": time_period,
                "values": {d: technical[d] for d in dates},
                "latest_rsi": list(technical.values())[0].get("RSI") if technical else None,
            }
            self._cache.set(key, result, ttl=3600)
            return result
        return {"ticker": ticker, "error": data.get("error", "No RSI data")}

    def get_macd(self, ticker: str, interval: str = "daily") -> dict[str, Any]:
        """MACD indicator."""
        key = self._cache.cache_key("av", "macd", ticker.upper(), interval)
        cached = self._cache.get(key)
        if cached:
            return cached

        data = self._call({
            "function": "MACD",
            "symbol": ticker,
            "interval": interval,
            "series_type": "close",
        })
        technical = data.get("Technical Analysis: MACD", {})
        if technical:
            dates = sorted(technical.keys(), reverse=True)[:3]
            result = {
                "ticker": ticker.upper(),
                "indicator": "MACD",
                "values": {d: technical[d] for d in dates},
            }
            self._cache.set(key, result, ttl=3600)
            return result
        return {"ticker": ticker, "error": data.get("error", "No MACD data")}

    def get_sector_performance(self) -> dict[str, Any]:
        """US market sector performance."""
        key = self._cache.cache_key("av", "sectors")
        cached = self._cache.get(key)
        if cached:
            return cached

        data = self._call({"function": "SECTOR"})
        if "Rank A: Real-Time Performance" in data:
            result = {
                "realtime": data.get("Rank A: Real-Time Performance", {}),
                "one_day": data.get("Rank B: 1 Day Performance", {}),
                "one_week": data.get("Rank C: 5 Day Performance", {}),
                "one_month": data.get("Rank D: 1 Month Performance", {}),
                "ytd": data.get("Rank E: Year-to-Date Performance", {}),
            }
            self._cache.set(key, result, ttl=1800)
            return result
        return {"error": data.get("error", "Sector data unavailable")}

    def get_income_statement(self, ticker: str) -> dict[str, Any]:
        """Annual income statement for the last 2 years."""
        key = self._cache.cache_key("av", "income", ticker.upper())
        cached = self._cache.get(key)
        if cached:
            return cached

        data = self._call({"function": "INCOME_STATEMENT", "symbol": ticker})
        reports = data.get("annualReports", [])[:2]
        if reports:
            self._cache.set(key, reports, ttl=86400)  # 24h
            return {"ticker": ticker.upper(), "annual_reports": reports}
        return {"ticker": ticker, "error": "No income statement data"}
