"""Tests for data clients with mocked HTTP calls."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.cache import CacheBackend


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
        mock_cache_inst.cache_key.return_value = "key"
        mock_cache.return_value = mock_cache_inst

        from src.data.fred_client import FredClient
        with patch("src.data.fred_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(fred_api_key="", apis=MagicMock(fred_base_url="https://api.stlouisfed.org/fred"))
            client = FredClient()
            result = client.get_series_latest("FEDFUNDS")

        assert "error" in result


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
