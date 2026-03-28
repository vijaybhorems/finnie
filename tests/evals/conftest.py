"""Shared fixtures for the eval suite.

These evals build a real FAISS index from the actual knowledge base
so they test the full retrieval pipeline — no mocks.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Environment variables needed by Settings (values don't matter for retrieval
# evals; LLM evals that hit the real API skip if the key is empty).
# ---------------------------------------------------------------------------
_REQUIRED_ENV = {
    "ANTHROPIC_API_KEY": "test_key",
    "ALPHA_VANTAGE_API_KEY": "test_key",
    "FRED_API_KEY": "test_key",
    "NEWS_API_KEY": "test_key",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
}


@pytest.fixture(autouse=True)
def _set_env_defaults(monkeypatch):
    """Ensure env vars exist so config loads without errors."""
    for key, default in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, os.environ.get(key, default))


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear LRU caches between tests."""
    yield
    from src.core.config import get_settings
    from src.core.llm import get_llm
    from src.rag.retriever import get_retriever
    from src.workflow.graph import build_graph

    get_settings.cache_clear()
    get_llm.cache_clear()
    get_retriever.cache_clear()
    build_graph.cache_clear()


# ---------------------------------------------------------------------------
# Shared fixture: a real RAG index built in a temp directory
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rag_index_dir():
    """Build a real FAISS index from the actual knowledge base once per session."""
    tmpdir = tempfile.mkdtemp(prefix="finnie_eval_rag_")

    # Patch settings to write index to tmpdir
    from unittest.mock import patch, MagicMock
    from src.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    # Build index into tmpdir
    from src.rag.indexer import RAGIndexer

    class _EvalIndexer(RAGIndexer):
        def __init__(self):
            super().__init__()
            self._index_path = Path(tmpdir)

    indexer = _EvalIndexer()
    indexer.build_index(force=True)

    yield Path(tmpdir)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture()
def retriever(rag_index_dir):
    """Return a RAGRetriever pointed at the session-scoped eval index."""
    from src.rag.retriever import RAGRetriever

    r = RAGRetriever()
    # Override index path to our eval index
    r._index_path = rag_index_dir
    # Force re-load
    r._index = None
    r._metadata = []
    r._embedder = None
    return r


# ---------------------------------------------------------------------------
# Helper: build a FinnieState for agent evals
# ---------------------------------------------------------------------------
@pytest.fixture()
def make_state():
    """Factory fixture to build a FinnieState for a given query."""
    from langchain_core.messages import HumanMessage
    from src.core.state import FinancialData, FinnieState, UserProfile

    def _factory(query: str, *, knowledge_level: str = "beginner", risk_tolerance: str = "moderate"):
        return FinnieState(
            messages=[HumanMessage(content=query)],
            user_profile=UserProfile(
                knowledge_level=knowledge_level,
                risk_tolerance=risk_tolerance,
            ),
            financial_data=FinancialData(),
        )

    return _factory
