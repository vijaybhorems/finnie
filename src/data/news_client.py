"""News client — NewsAPI, financial RSS feeds and SEC EDGAR filings."""
from __future__ import annotations

from typing import Any

import feedparser
import requests

from src.core.config import get_settings
from src.utils.cache import get_cache
from src.utils.circuit_breaker import get_breaker
from src.utils.logger import get_logger
from src.utils.parallel import gather
from src.utils.redact import redact

logger = get_logger(__name__)

_NEWSAPI_BREAKER = "newsapi"
_RSS_BREAKER = "rss"
_SEC_BREAKER = "sec_edgar"

SEC_EDGAR_RSS = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&output=atom"

# Checked 2026-11-23. Reuters' feeds.reuters.com no longer resolves and Yahoo's
# rssindex returns 404: both were here, and silently returned nothing.
FINANCIAL_RSS_FEEDS = {
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
}
_FEED_TIMEOUT_SECONDS = 8
_FEED_USER_AGENT = "Mozilla/5.0 (compatible; Finnie/1.0; finance education)"


def fetch_feed(url: str, headers: dict[str, str] | None = None) -> list[Any]:
    """A feed's entries; raises when the feed can't be fetched or read.

    `feedparser.parse(url)` never raises: a dead host, a 404 or a 403 all come
    back as an empty feed, so a broken source looked like a quiet one and its
    circuit breaker counted a success. It also has no timeout.
    """
    resp = requests.get(
        url,
        headers=headers or {"User-Agent": _FEED_USER_AGENT},
        timeout=_FEED_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    if not feed.entries and feed.bozo:
        raise ValueError(f"unreadable feed: {feed.get('bozo_exception')}")
    return list(feed.entries)


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

        breaker = get_breaker(_NEWSAPI_BREAKER)
        if not breaker.allow():
            return self._get_rss_headlines()

        try:
            resp = requests.get(
                f"{self._base_url}/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": str(page_size),
                    "domains": "reuters.com,bloomberg.com,cnbc.com,wsj.com,ft.com",
                },
                headers={"X-Api-Key": self._api_key},  # a header, so the key is never in a URL
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
            breaker.record_success()
            self._cache.set(key, results, ttl=1800)  # 30 min
            return results
        except Exception as exc:
            breaker.record_failure()
            logger.error("newsapi_error", error=redact(str(exc), [self._api_key]))
            return self._get_rss_headlines()

    def _get_rss_headlines(self, max_per_feed: int = 5) -> list[dict[str, Any]]:
        """Fallback: the latest items from general financial RSS feeds."""
        key = self._cache.cache_key("news", "rss")
        cached = self._cache.get(key)
        if cached:
            return cached

        breaker = get_breaker(_RSS_BREAKER)
        if not breaker.allow():
            stale = self._cache.get_stale(key)
            if stale is not None:
                logger.warning("served_stale_cache", provider=_RSS_BREAKER, key=key)
                return stale
            return []

        fetched = gather(
            {source: (lambda u=url: fetch_feed(u)) for source, url in FINANCIAL_RSS_FEEDS.items()},
            timeout=_FEED_TIMEOUT_SECONDS + 2,
        )
        results: list[dict[str, Any]] = []
        for source in FINANCIAL_RSS_FEEDS:
            entries = fetched.get(source)
            if entries is None:
                logger.warning("rss_feed_error", source=source)
                continue
            for entry in entries[:max_per_feed]:
                results.append({
                    "title": entry.get("title", ""),
                    "source": source,
                    "url": entry.get("link", ""),
                    "published_at": entry.get("published", ""),
                    "description": entry.get("summary", "")[:300],
                })

        if not results:
            breaker.record_failure()
            stale = self._cache.get_stale(key)
            if stale is not None:
                logger.warning("served_stale_cache", provider=_RSS_BREAKER, key=key)
                return stale
            return []

        breaker.record_success()
        self._cache.set(key, results, ttl=1800)
        return results

    def get_ticker_news(self, ticker: str, page_size: int = 5) -> list[dict[str, Any]]:
        """Get news specifically about a ticker symbol."""
        return self.get_financial_headlines(
            query=f"{ticker} stock earnings",
            page_size=page_size,
        )

    def get_sec_filings(self, max_items: int = 10) -> list[dict[str, Any]]:
        """Parse latest 8-K filings from SEC EDGAR RSS.

        SEC answers 403 to a User-Agent without a contact email, so this needs
        SEC_USER_AGENT ("Finnie admin@example.com"); without it there are no filings.
        """
        user_agent = self._settings.sec_user_agent
        if not user_agent:
            logger.info("sec_edgar_skipped", reason="SEC_USER_AGENT not set")
            return []

        key = self._cache.cache_key("news", "sec_edgar")
        cached = self._cache.get(key)
        if cached:
            return cached

        breaker = get_breaker(_SEC_BREAKER)
        if not breaker.allow():
            stale = self._cache.get_stale(key)
            if stale is not None:
                logger.warning("served_stale_cache", provider=_SEC_BREAKER, key=key)
                return stale
            return []

        try:
            entries = fetch_feed(SEC_EDGAR_RSS, headers={"User-Agent": user_agent})
            results = [
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", "")[:300],
                }
                for entry in entries[:max_items]
            ]
            breaker.record_success()
            self._cache.set(key, results, ttl=3600)
            return results
        except Exception as exc:
            breaker.record_failure()
            logger.error("sec_edgar_error", error=str(exc))
            stale = self._cache.get_stale(key)
            if stale is not None:
                logger.warning("served_stale_cache", provider=_SEC_BREAKER, key=key)
                return stale
            return []
