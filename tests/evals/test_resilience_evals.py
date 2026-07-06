"""Resilience evals — circuit breaker + stale-cache fallback end-to-end.

Exercises a real data client (YFinanceClient) through repeated upstream
failures to confirm that:
  1. failures are served from the stale cache mirror, and
  2. once the breaker opens, further calls short-circuit (no network attempts)
     while still serving stale data.

Run with:
    pytest tests/evals/test_resilience_evals.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.config import get_settings
from src.utils.circuit_breaker import CircuitState, get_breaker, reset_breakers


def _mock_cache(stale_value):
    cache = MagicMock()
    cache.cache_key.return_value = "yf:price:AAPL"
    cache.get.return_value = None          # no fresh entry
    cache.get_stale.return_value = stale_value  # stale mirror available
    return cache


class TestYFinanceResilience:
    """Breaker opens after threshold failures; stale cache keeps serving."""

    def test_breaker_opens_and_serves_stale(self):
        reset_breakers()
        threshold = get_settings().circuit_breaker.failure_threshold
        stale = {"ticker": "AAPL", "current_price": 123.0, "_stale": True}

        with patch("src.data.yfinance_client.get_cache", return_value=_mock_cache(stale)), \
             patch("src.data.yfinance_client.yf.Ticker", side_effect=RuntimeError("upstream down")) as mock_ticker:
            from src.data.yfinance_client import YFinanceClient
            client = YFinanceClient()

            # Drive failures up to the threshold — each returns stale, hits network.
            for _ in range(threshold):
                assert client.get_current_price("AAPL") == stale

            # Breaker should now be OPEN.
            assert get_breaker("yfinance").state == CircuitState.OPEN
            network_calls_after_trip = mock_ticker.call_count

            # Further calls short-circuit: still stale, but NO new network attempts.
            for _ in range(3):
                assert client.get_current_price("AAPL") == stale
            assert mock_ticker.call_count == network_calls_after_trip, (
                "Open breaker must not attempt further network calls"
            )

    def test_no_stale_returns_error_payload(self):
        """With no stale mirror, an open breaker returns the circuit_open error payload."""
        reset_breakers()
        threshold = get_settings().circuit_breaker.failure_threshold

        with patch("src.data.yfinance_client.get_cache", return_value=_mock_cache(None)), \
             patch("src.data.yfinance_client.yf.Ticker", side_effect=RuntimeError("upstream down")):
            from src.data.yfinance_client import YFinanceClient
            client = YFinanceClient()
            for _ in range(threshold):
                client.get_current_price("AAPL")

            result = client.get_current_price("AAPL")
            assert result.get("error") == "circuit_open"
