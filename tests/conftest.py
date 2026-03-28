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


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Clear LRU-cached singletons between tests to prevent state leakage."""
    yield
    from src.core.config import get_settings
    from src.core.llm import get_llm
    from src.workflow.graph import build_graph
    get_settings.cache_clear()
    get_llm.cache_clear()
    build_graph.cache_clear()
