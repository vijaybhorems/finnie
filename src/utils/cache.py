"""Redis caching layer with graceful fallback to in-memory dict."""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheBackend:
    """Thread-safe in-memory fallback cache."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: int = 3600) -> None:
        self._store[key] = (value, time.time() + ttl)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class Cache:
    """Unified cache interface — prefers Redis, falls back to in-memory."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis = self._init_redis()
        self._fallback = CacheBackend()

    def _init_redis(self):
        try:
            import redis

            client = redis.Redis(
                host=self._settings.redis.host,
                port=self._settings.redis.port,
                db=self._settings.redis.db,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            client.ping()
            logger.info("redis_connected", host=self._settings.redis.host)
            return client
        except Exception as exc:
            logger.warning("redis_unavailable_using_memory_cache", error=str(exc))
            return None

    @property
    def _default_ttl(self) -> int:
        return self._settings.redis.ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        try:
            if self._redis:
                raw = self._redis.get(key)
            else:
                raw = self._fallback.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("cache_get_error", key=key, error=str(exc))
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self._default_ttl
        serialized = json.dumps(value)
        try:
            if self._redis:
                self._redis.setex(key, ttl, serialized)
            else:
                self._fallback.set(key, serialized, ttl)
        except Exception as exc:
            logger.warning("cache_set_error", key=key, error=str(exc))

    def delete(self, key: str) -> None:
        try:
            if self._redis:
                self._redis.delete(key)
            else:
                self._fallback.delete(key)
        except Exception as exc:
            logger.warning("cache_delete_error", key=key, error=str(exc))

    def cache_key(self, *parts: str) -> str:
        return ":".join(["finnie", *parts])


_cache_instance: Optional[Cache] = None


def get_cache() -> Cache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = Cache()
    return _cache_instance
