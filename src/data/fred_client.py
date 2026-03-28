"""FRED (Federal Reserve Economic Data) client."""
from __future__ import annotations

from typing import Any

import requests

from src.core.config import get_settings
from src.utils.cache import get_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Key FRED series IDs used across agents
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "gdp_growth": "A191RL1Q225SBEA",
    "unemployment": "UNRATE",
    "inflation_expectations": "T5YIE",
    "10yr_treasury": "GS10",
    "30yr_mortgage": "MORTGAGE30US",
    "sp500_pe_ratio": "MULTPL/SHILLER_PE_RATIO_MONTH",
    "vix": "VIXCLS",
    "m2_money_supply": "M2SL",
}


class FredClient:
    """Wraps the FRED REST API with caching."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cache = get_cache()
        self._api_key = self._settings.fred_api_key
        self._base_url = self._settings.apis.fred_base_url

    def _call(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        if not self._api_key:
            return {"error": "FRED API key not configured"}
        params["api_key"] = self._api_key
        params["file_type"] = "json"
        try:
            resp = requests.get(f"{self._base_url}/{endpoint}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("fred_request_error", endpoint=endpoint, error=str(exc))
            return {"error": str(exc)}

    def get_series_latest(self, series_id: str) -> dict[str, Any]:
        """Return the most recent observation for a series."""
        key = self._cache.cache_key("fred", "latest", series_id)
        cached = self._cache.get(key)
        if cached:
            return cached

        data = self._call("series/observations", {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": "1",
        })
        observations = data.get("observations", [])
        if observations:
            result = {
                "series_id": series_id,
                "date": observations[0]["date"],
                "value": observations[0]["value"],
            }
            self._cache.set(key, result, ttl=3600)
            return result
        return {"series_id": series_id, "error": data.get("error", "No data")}

    def get_series_history(self, series_id: str, limit: int = 24) -> list[dict[str, Any]]:
        """Return the last N observations for a series."""
        key = self._cache.cache_key("fred", "history", series_id, str(limit))
        cached = self._cache.get(key)
        if cached:
            return cached

        data = self._call("series/observations", {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": str(limit),
        })
        observations = data.get("observations", [])
        if observations:
            result = [{"date": o["date"], "value": o["value"]} for o in observations]
            self._cache.set(key, result, ttl=3600)
            return result
        return []

    def get_macro_snapshot(self) -> dict[str, Any]:
        """Return latest values for key macro indicators."""
        key = self._cache.cache_key("fred", "macro_snapshot")
        cached = self._cache.get(key)
        if cached:
            return cached

        snapshot: dict[str, Any] = {}
        for name, series_id in FRED_SERIES.items():
            result = self.get_series_latest(series_id)
            if "error" not in result:
                snapshot[name] = {"value": result["value"], "date": result["date"]}
            else:
                snapshot[name] = {"error": result["error"]}

        self._cache.set(key, snapshot, ttl=3600)
        return snapshot

    def get_interest_rate_environment(self) -> dict[str, Any]:
        """Summarise the current rate environment for goal planning."""
        key = self._cache.cache_key("fred", "rate_env")
        cached = self._cache.get(key)
        if cached:
            return cached

        fed_funds = self.get_series_latest(FRED_SERIES["fed_funds_rate"])
        treasury_10yr = self.get_series_latest(FRED_SERIES["10yr_treasury"])
        cpi = self.get_series_latest(FRED_SERIES["cpi"])
        inflation_exp = self.get_series_latest(FRED_SERIES["inflation_expectations"])
        mortgage = self.get_series_latest(FRED_SERIES["30yr_mortgage"])

        result = {
            "fed_funds_rate": fed_funds.get("value"),
            "fed_funds_date": fed_funds.get("date"),
            "treasury_10yr": treasury_10yr.get("value"),
            "cpi": cpi.get("value"),
            "inflation_expectations_5yr": inflation_exp.get("value"),
            "mortgage_30yr": mortgage.get("value"),
        }
        self._cache.set(key, result, ttl=3600)
        return result
