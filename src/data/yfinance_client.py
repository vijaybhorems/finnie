"""yFinance client — historical prices, fundamentals, dividends, beta."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from src.utils.cache import get_cache
from src.utils.circuit_breaker import get_breaker
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BREAKER_NAME = "yfinance"

# SPDR sector ETFs — used to derive sector performance since Alpha Vantage's
# free SECTOR endpoint was deprecated (it now returns an empty payload).
_SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


class YFinanceClient:
    """Wraps yfinance with Redis caching and error handling."""

    def __init__(self) -> None:
        self._cache = get_cache()

    def _stale_or(self, key: str, default: Any) -> Any:
        """Return the stale cached value for `key` if present, else `default`."""
        stale = self._cache.get_stale(key)
        if stale is not None:
            logger.warning("served_stale_cache", provider=_BREAKER_NAME, key=key)
            return stale
        return default

    def get_current_price(self, ticker: str) -> dict[str, Any]:
        """Return latest price, change%, volume for a ticker."""
        key = self._cache.cache_key("yf", "price", ticker.upper())
        cached = self._cache.get(key)
        if cached:
            return cached

        breaker = get_breaker(_BREAKER_NAME)
        if not breaker.allow():
            return self._stale_or(key, {"ticker": ticker, "error": "circuit_open"})

        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            data = {
                "ticker": ticker.upper(),
                "current_price": getattr(info, "last_price", None),
                "previous_close": getattr(info, "previous_close", None),
                "day_high": getattr(info, "day_high", None),
                "day_low": getattr(info, "day_low", None),
                "volume": getattr(info, "three_month_average_volume", None),
                "market_cap": getattr(info, "market_cap", None),
                "currency": getattr(info, "currency", "USD"),
            }
            if data["current_price"] and data["previous_close"]:
                data["change_pct"] = round(
                    (data["current_price"] - data["previous_close"]) / data["previous_close"] * 100, 2
                )
            else:
                data["change_pct"] = None
            breaker.record_success()
            self._cache.set(key, data, ttl=300)  # 5-min cache for prices
            return data
        except Exception as exc:
            breaker.record_failure()
            logger.error("yfinance_price_error", ticker=ticker, error=str(exc))
            return self._stale_or(key, {"ticker": ticker, "error": str(exc)})

    def get_current_prices(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch current price for many tickers concurrently.

        Returns ``{TICKER: price_dict}`` keyed by upper-cased ticker. Each ticker
        is fetched through :meth:`get_current_price`, so caching, the circuit
        breaker, and per-ticker error handling are unchanged — this only replaces
        sequential network round-trips with parallel ones.
        """
        # De-duplicate while preserving order (uppercase to match get_current_price).
        unique = list(dict.fromkeys(t.upper() for t in tickers if t and t.strip()))
        if not unique:
            return {}

        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(len(unique), 8)) as executor:
            future_to_ticker = {
                executor.submit(self.get_current_price, ticker): ticker
                for ticker in unique
            }
            for future, ticker in future_to_ticker.items():
                try:
                    results[ticker] = future.result()
                except Exception as exc:  # get_current_price already guards, but be safe
                    logger.error("yfinance_price_error", ticker=ticker, error=str(exc))
                    results[ticker] = {"ticker": ticker, "error": str(exc)}
        return results

    def get_sector_performance(self) -> dict[str, Any]:
        """US sector performance (1-day % change) derived from SPDR sector ETFs.

        Alpha Vantage's free ``SECTOR`` endpoint was deprecated (it now returns an
        empty payload), so sector moves are computed from the SPDR sector ETFs via
        yfinance. The return shape mirrors the previous provider so callers are
        unchanged: ``{"one_day": {sector_name: "+x.xx%"}}`` (or ``{"error": ...}``).
        """
        key = self._cache.cache_key("yf", "sectors")
        cached = self._cache.get(key)
        if cached:
            return cached

        # Fetch all sector ETFs in parallel, then build the ordered result.
        prices = self.get_current_prices(list(_SECTOR_ETFS.values()))
        one_day: dict[str, str] = {}
        for sector, etf in _SECTOR_ETFS.items():
            change_pct = prices.get(etf.upper(), {}).get("change_pct")
            if change_pct is not None:
                one_day[sector] = f"{change_pct:+.2f}%"

        if not one_day:
            return self._stale_or(key, {"error": "Sector data unavailable"})

        result = {"one_day": one_day}
        self._cache.set(key, result, ttl=1800)  # 30-min cache
        return result

    def get_historical_prices(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[dict[str, Any]]:
        """Return OHLCV history as a list of dicts."""
        key = self._cache.cache_key("yf", "history", ticker.upper(), period, interval)
        cached = self._cache.get(key)
        if cached:
            return cached

        breaker = get_breaker(_BREAKER_NAME)
        if not breaker.allow():
            return self._stale_or(key, [])

        try:
            df: pd.DataFrame | None = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                return []
            df.reset_index(inplace=True)
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(c).strip("_") for c in df.columns]
            records = df.to_dict(orient="records")
            # Convert timestamps to ISO strings
            for r in records:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
            breaker.record_success()
            self._cache.set(key, records, ttl=3600)
            return records
        except Exception as exc:
            breaker.record_failure()
            logger.error("yfinance_history_error", ticker=ticker, error=str(exc))
            return self._stale_or(key, [])

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Return key fundamental metrics (P/E, beta, dividends, etc.)."""
        key = self._cache.cache_key("yf", "fundamentals", ticker.upper())
        cached = self._cache.get(key)
        if cached:
            return cached

        breaker = get_breaker(_BREAKER_NAME)
        if not breaker.allow():
            return self._stale_or(key, {"ticker": ticker, "error": "circuit_open"})

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fields = [
                "trailingPE", "forwardPE", "priceToBook", "beta",
                "dividendYield", "payoutRatio", "trailingEps", "forwardEps",
                "returnOnEquity", "returnOnAssets", "debtToEquity",
                "currentRatio", "quickRatio", "grossMargins", "operatingMargins",
                "revenueGrowth", "earningsGrowth", "52WeekChange",
                "shortName", "sector", "industry", "country",
            ]
            data = {f: info.get(f) for f in fields}
            data["ticker"] = ticker.upper()
            breaker.record_success()
            self._cache.set(key, data, ttl=3600)
            return data
        except Exception as exc:
            breaker.record_failure()
            logger.error("yfinance_fundamentals_error", ticker=ticker, error=str(exc))
            return self._stale_or(key, {"ticker": ticker, "error": str(exc)})

    def get_portfolio_metrics(
        self, holdings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Compute portfolio-level metrics for a list of holdings.
        holdings: [{"ticker": "AAPL", "shares": 10, "avg_cost": 150.0}, ...]
        """
        results: list[dict[str, Any]] = []
        total_value = 0.0
        total_cost = 0.0

        for h in holdings:
            ticker = h["ticker"]
            shares = float(h.get("shares", 0))
            avg_cost = float(h.get("avg_cost", 0))

            price_data = self.get_current_price(ticker)
            current_price = price_data.get("current_price") or 0.0
            position_value = current_price * shares
            position_cost = avg_cost * shares
            gain_loss = position_value - position_cost
            gain_loss_pct = (gain_loss / position_cost * 100) if position_cost > 0 else 0.0

            fundamentals = self.get_fundamentals(ticker)

            results.append({
                "ticker": ticker.upper(),
                "shares": shares,
                "avg_cost": avg_cost,
                "current_price": current_price,
                "position_value": round(position_value, 2),
                "gain_loss": round(gain_loss, 2),
                "gain_loss_pct": round(gain_loss_pct, 2),
                "beta": fundamentals.get("beta"),
                "pe_ratio": fundamentals.get("trailingPE"),
                "sector": fundamentals.get("sector"),
                "dividend_yield": fundamentals.get("dividendYield"),
            })

            total_value += position_value
            total_cost += position_cost

        total_gain_loss = total_value - total_cost
        return {
            "holdings": results,
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain_loss": round(total_gain_loss, 2),
            "total_gain_loss_pct": round(total_gain_loss / total_cost * 100, 2) if total_cost > 0 else 0.0,
        }
