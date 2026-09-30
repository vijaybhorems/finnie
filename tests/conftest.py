"""Shared pytest fixtures and configuration."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set minimal env vars so settings load without real API keys in tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_av_key")
    monkeypatch.setenv("FRED_API_KEY", "test_fred_key")
    monkeypatch.setenv("NEWS_API_KEY", "test_news_key")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    # Tests use the in-memory persistence backend. Set explicitly (not just
    # unset): pydantic-settings would otherwise read DATABASE_URL from a
    # developer's .env and point the suite at a real database.
    monkeypatch.setenv("DATABASE_URL", "")


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Clear LRU-cached singletons between tests to prevent state leakage."""
    yield
    import src.utils.cache as cache_module
    from src.core.config import get_settings
    from src.core.embeddings import get_embedder
    from src.core.llm import get_llm
    from src.rag.version import get_kb_version
    from src.utils.circuit_breaker import reset_breakers
    from src.utils.semantic_cache import get_faq_cache
    from src.workflow.graph import build_graph
    get_settings.cache_clear()
    get_llm.cache_clear()
    build_graph.cache_clear()
    get_faq_cache.cache_clear()
    get_kb_version.cache_clear()
    get_embedder.cache_clear()
    # The cache singleton holds an in-memory fallback dict that would otherwise
    # leak entries (and FAQ answers) between tests.
    cache_module._cache_instance = None
    reset_breakers()
    from src.persistence.backend import reset_persistence
    reset_persistence()
    from src.rag.pgvector import ensure_schema
    from src.rag.retriever import get_retriever
    ensure_schema.cache_clear()
    get_retriever.cache_clear()
