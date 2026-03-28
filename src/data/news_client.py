"""News client — NewsAPI + SEC EDGAR RSS feed."""
from __future__ import annotations

from typing import Any

import feedparser
import requests

from src.core.config import get_settings
from src.utils.cache import get_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEC_EDGAR_RSS = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&output=atom"

FINANCIAL_RSS_FEEDS = {
    "Reuters Markets": "https://feeds.reuters.com/reuters/businessNews",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
}


class NewsClient:
    """Aggregates financial news from NewsAPI and RSS feeds."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cache = get_cache()
        self._api_key = self._settings.news_api_key
        self._base_url = self._settings.apis.news_api_base_url

    def get_financial_headlines(
        self,
        query: str = "stock market finance investing",
        page_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch top financial headlines from NewsAPI."""
        key = self._cache.cache_key("news", "headlines", query[:50])
        cached = self._cache.get(key)
        if cached:
            return cached

        if not self._api_key:
            logger.warning("newsapi_key_not_configured_falling_back_to_rss")
            return self._get_rss_headlines()

        try:
            resp = requests.get(
                f"{self._base_url}/everything",
                params={
                    "q": query,
                    "apiKey": self._api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": str(page_size),
                    "domains": "reuters.com,bloomberg.com,cnbc.com,wsj.com,ft.com",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            results = [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                    "description": a.get("description", ""),
                }
                for a in articles
            ]
            self._cache.set(key, results, ttl=1800)  # 30 min
            return results
        except Exception as exc:
            logger.error("newsapi_error", error=str(exc))
            return self._get_rss_headlines()

    def _get_rss_headlines(self, max_per_feed: int = 5) -> list[dict[str, Any]]:
        """Fallback: parse RSS feeds."""
        key = self._cache.cache_key("news", "rss")
        cached = self._cache.get(key)
        if cached:
            return cached

        results: list[dict[str, Any]] = []
        for source, url in FINANCIAL_RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_per_feed]:
                    results.append({
                        "title": entry.get("title", ""),
                        "source": source,
                        "url": entry.get("link", ""),
                        "published_at": entry.get("published", ""),
                        "description": entry.get("summary", "")[:300],
                    })
            except Exception as exc:
                logger.warning("rss_feed_error", source=source, error=str(exc))

        self._cache.set(key, results, ttl=1800)
        return results

    def get_ticker_news(self, ticker: str, page_size: int = 5) -> list[dict[str, Any]]:
        """Get news specifically about a ticker symbol."""
        return self.get_financial_headlines(
            query=f"{ticker} stock earnings",
            page_size=page_size,
        )

    def get_sec_filings(self, max_items: int = 10) -> list[dict[str, Any]]:
        """Parse latest 8-K filings from SEC EDGAR RSS."""
        key = self._cache.cache_key("news", "sec_edgar")
        cached = self._cache.get(key)
        if cached:
            return cached

        try:
            feed = feedparser.parse(SEC_EDGAR_RSS)
            results = [
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", "")[:300],
                }
                for entry in feed.entries[:max_items]
            ]
            self._cache.set(key, results, ttl=3600)
            return results
        except Exception as exc:
            logger.error("sec_edgar_error", error=str(exc))
            return []
