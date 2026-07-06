"""Tests for data clients with mocked HTTP calls."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.cache import CacheBackend
from src.utils.circuit_breaker import CircuitBreaker, CircuitState


class TestCacheBackend:
    def test_set_and_get(self):
        cache = CacheBackend()
        cache.set("key1", "value1", ttl=60)
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        cache = CacheBackend()
        assert cache.get("nonexistent") is None

    def test_expired_key_returns_none(self):
        import time
        cache = CacheBackend()
        cache.set("key2", "value2", ttl=1)
        time.sleep(1.1)
        assert cache.get("key2") is None

    def test_delete(self):
        cache = CacheBackend()
        cache.set("key3", "value3", ttl=60)
        cache.delete("key3")
        assert cache.get("key3") is None

    def test_exists(self):
        cache = CacheBackend()
        cache.set("key4", "value4", ttl=60)
        assert cache.exists("key4") is True
        assert cache.exists("nonexistent") is False


class TestUnifiedCache:
    @patch("src.utils.cache.redis")
    def test_uses_fallback_when_redis_unavailable(self, mock_redis_module):
        mock_redis_module.Redis.side_effect = Exception("Redis unavailable")
        from src.utils.cache import Cache
        cache = Cache()
        assert cache._redis is None

        cache.set("test_key", {"value": 42})
        result = cache.get("test_key")
        assert result == {"value": 42}

    def test_cache_key_format(self):
        from src.utils.cache import Cache
        with patch.object(Cache, "_init_redis", return_value=None):
            cache = Cache()
        key = cache.cache_key("yf", "price", "AAPL")
        assert key == "finnie:yf:price:AAPL"

    def test_get_stale_returns_value_after_fresh_expiry(self):
        import time
        from src.utils.cache import Cache
        with patch.object(Cache, "_init_redis", return_value=None):
            cache = Cache()
        key = cache.cache_key("yf", "price", "AAPL")
        cache.set(key, {"price": 100}, ttl=1)

        # Fresh copy expires, but the stale mirror (7d) persists.
        time.sleep(1.1)
        assert cache.get(key) is None
        assert cache.get_stale(key) == {"price": 100}


class TestCircuitBreaker:
    def _breaker(self, **kwargs):
        defaults = dict(name="test", failure_threshold=3, recovery_timeout=60, success_threshold=1)
        defaults.update(kwargs)
        return CircuitBreaker(**defaults)

    def test_opens_after_threshold_failures(self):
        cb = self._breaker(failure_threshold=3)
        assert cb.allow() is True
        for _ in range(3):
            cb.record_failure()
        assert cb.state is CircuitState.OPEN
        assert cb.allow() is False

    def test_half_open_after_recovery_timeout(self):
        cb = self._breaker(failure_threshold=1, recovery_timeout=30)
        cb.record_failure()
        assert cb.allow() is False
        with patch("src.utils.circuit_breaker.time.time", return_value=cb._opened_at + 31):
            assert cb.allow() is True
            assert cb.state is CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = self._breaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        assert cb.allow() is True  # transitions to HALF_OPEN
        cb.record_success()
        assert cb.state is CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = self._breaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        assert cb.allow() is True  # HALF_OPEN
        cb.record_failure()
        assert cb.state is CircuitState.OPEN


class TestYFinanceClient:
    @patch("src.data.yfinance_client.yf")
    @patch("src.data.yfinance_client.get_cache")
    def test_get_current_price_cached(self, mock_cache, mock_yf):
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.return_value = {"ticker": "AAPL", "current_price": 200.0}
        mock_cache.return_value = mock_cache_inst

        from src.data.yfinance_client import YFinanceClient
        client = YFinanceClient()
        result = client.get_current_price("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["current_price"] == 200.0
        mock_yf.Ticker.assert_not_called()

    def test_get_portfolio_metrics_empty(self):
        from src.data.yfinance_client import YFinanceClient
        with patch("src.data.yfinance_client.get_cache") as mock_cache:
            mock_cache.return_value = MagicMock(get=MagicMock(return_value=None), set=MagicMock(), cache_key=MagicMock(return_value="key"))
            with patch("src.data.yfinance_client.yf") as mock_yf:
                mock_yf.Ticker.return_value.fast_info = MagicMock(last_price=None, previous_close=None, day_high=None, day_low=None, three_month_average_volume=None, market_cap=None, currency="USD")
                mock_yf.Ticker.return_value.info = {}
                mock_yf.download.return_value = __import__("pandas").DataFrame()
                client = YFinanceClient()
                result = client.get_portfolio_metrics([])

        assert result["total_value"] == 0.0
        assert result["holdings"] == []


class TestFredClient:
    @patch("src.data.fred_client.requests")
    @patch("src.data.fred_client.get_cache")
    def test_get_series_latest_cached(self, mock_cache, mock_requests):
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.return_value = {"series_id": "FEDFUNDS", "date": "2024-01-01", "value": "5.33"}
        mock_cache.return_value = mock_cache_inst

        from src.data.fred_client import FredClient
        client = FredClient()
        result = client.get_series_latest("FEDFUNDS")

        assert result["value"] == "5.33"
        mock_requests.get.assert_not_called()

    @patch("src.data.fred_client.requests")
    @patch("src.data.fred_client.get_cache")
    def test_get_series_latest_no_api_key(self, mock_cache, mock_requests):
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.return_value = None
        mock_cache_inst.get_stale.return_value = None
        mock_cache_inst.cache_key.return_value = "key"
        mock_cache.return_value = mock_cache_inst

        from src.data.fred_client import FredClient
        with patch("src.data.fred_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fred_api_key="", apis=MagicMock(fred_base_url="https://api.stlouisfed.org/fred"))
            client = FredClient()
            result = client.get_series_latest("FEDFUNDS")

        assert "error" in result


class TestAlphaVantageCircuitBreaker:
    @patch("src.data.alpha_vantage_client.time")
    @patch("src.data.alpha_vantage_client.requests")
    @patch("src.data.alpha_vantage_client.get_cache")
    def test_open_breaker_skips_sleep_and_network(self, mock_cache, mock_requests, mock_time):
        from src.data.alpha_vantage_client import AlphaVantageClient
        from src.utils.circuit_breaker import get_breaker

        mock_cache.return_value = MagicMock(get=MagicMock(return_value=None), cache_key=MagicMock(return_value="finnie:av:quote:AAPL"))

        client = AlphaVantageClient()
        # Trip the breaker.
        breaker = get_breaker("alpha_vantage")
        for _ in range(breaker._failure_threshold):
            breaker.record_failure()

        result = client._call({"function": "GLOBAL_QUOTE", "symbol": "AAPL"})

        assert result == {"error": "circuit_open"}
        mock_time.sleep.assert_not_called()
        mock_requests.get.assert_not_called()


class TestYFinanceStaleFallback:
    @patch("src.data.yfinance_client.yf")
    @patch("src.data.yfinance_client.get_cache")
    def test_open_breaker_serves_stale(self, mock_cache, mock_yf):
        from src.data.yfinance_client import YFinanceClient
        from src.utils.circuit_breaker import get_breaker

        stale_value = {"ticker": "AAPL", "current_price": 199.0}
        mock_cache.return_value = MagicMock(
            get=MagicMock(return_value=None),
            get_stale=MagicMock(return_value=stale_value),
            cache_key=MagicMock(return_value="finnie:yf:price:AAPL"),
        )

        breaker = get_breaker("yfinance")
        for _ in range(breaker._failure_threshold):
            breaker.record_failure()

        client = YFinanceClient()
        result = client.get_current_price("AAPL")

        assert result == stale_value
        mock_yf.Ticker.assert_not_called()

    @patch("src.data.yfinance_client.yf")
    @patch("src.data.yfinance_client.get_cache")
    def test_open_breaker_error_when_no_stale(self, mock_cache, mock_yf):
        from src.data.yfinance_client import YFinanceClient
        from src.utils.circuit_breaker import get_breaker

        mock_cache.return_value = MagicMock(
            get=MagicMock(return_value=None),
            get_stale=MagicMock(return_value=None),
            cache_key=MagicMock(return_value="finnie:yf:price:AAPL"),
        )

        breaker = get_breaker("yfinance")
        for _ in range(breaker._failure_threshold):
            breaker.record_failure()

        client = YFinanceClient()
        result = client.get_current_price("AAPL")

        assert result["error"] == "circuit_open"
        mock_yf.Ticker.assert_not_called()


class TestFredStaleFallback:
    @patch("src.data.fred_client.requests")
    @patch("src.data.fred_client.get_cache")
    def test_failure_serves_stale(self, mock_cache, mock_requests):
        from src.data.fred_client import FredClient

        stale_value = {"series_id": "FEDFUNDS", "date": "2024-01-01", "value": "5.33"}
        mock_cache.return_value = MagicMock(
            get=MagicMock(return_value=None),
            get_stale=MagicMock(return_value=stale_value),
            cache_key=MagicMock(return_value="finnie:fred:latest:FEDFUNDS"),
        )
        mock_requests.get.side_effect = Exception("network down")

        client = FredClient()
        result = client.get_series_latest("FEDFUNDS")

        assert result == stale_value


class TestNewsClient:
    @patch("src.data.news_client.feedparser")
    @patch("src.data.news_client.get_cache")
    def test_get_rss_headlines_fallback(self, mock_cache, mock_feedparser):
        mock_cache_inst = MagicMock()
        mock_cache_inst.get.return_value = None
        mock_cache_inst.cache_key.return_value = "key"
        mock_cache.return_value = mock_cache_inst

        mock_feed = MagicMock()
        mock_feed.entries = [
            MagicMock(title="Test headline", link="https://example.com", published="2024-01-01", summary="Test summary"),
        ]
        mock_feedparser.parse.return_value = mock_feed

        from src.data.news_client import NewsClient
        client = NewsClient()
        results = client._get_rss_headlines()

        assert len(results) > 0
        assert results[0]["title"] == "Test headline"
