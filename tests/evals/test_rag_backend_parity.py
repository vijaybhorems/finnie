"""Retrieval parity: pgvector hybrid search vs the FAISS index.

Phase 4's exit criterion is recall no worse than FAISS. With a 20-chunk
corpus, "expected category somewhere in the top 5" barely separates the two
(top-5 is a quarter of the corpus), so this also scores MRR@5 — how *high*
the expected category ranks — which is where hybrid ranking can help or hurt.

Real embedding model, real knowledge base, real Postgres:

    FINNIE_TEST_DATABASE_URL=postgresql://user@localhost:5432/finnie_test \\
        pytest tests/evals/test_rag_backend_parity.py -v -s
"""
from __future__ import annotations

import os

import pytest

from tests.evals.test_rag_retrieval import RETRIEVAL_CASES

_PG_URL = os.environ.get("FINNIE_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _PG_URL, reason="set FINNIE_TEST_DATABASE_URL")


@pytest.fixture
def pgvector_retriever(monkeypatch):
    from src.core.config import get_settings
    from src.persistence.backend import reset_persistence
    from src.rag.pgvector import PgVectorRetriever, ensure_schema, sync_knowledge_base

    monkeypatch.setenv("DATABASE_URL", _PG_URL)
    get_settings.cache_clear()
    ensure_schema.cache_clear()
    reset_persistence()
    sync_knowledge_base()           # the real knowledge base, real model
    yield PgVectorRetriever()
    reset_persistence()


def _score(retriever) -> dict[str, float]:
    category_hits = keyword_hits = 0
    reciprocal_ranks = []
    for query, expected_category, keywords in RETRIEVAL_CASES:
        results = retriever.search(query, top_k=5)
        categories = [r["category"] for r in results]
        text = " ".join(r["text"] for r in results).lower()
        category_hits += expected_category in categories
        keyword_hits += any(k.lower() in text for k in keywords)
        rank = categories.index(expected_category) + 1 if expected_category in categories else None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return {
        "category_hit@5": category_hits,
        "keyword_hit@5": keyword_hits,
        "mrr@5": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "top1": sum(1 for r in reciprocal_ranks if r == 1.0),
    }


def test_pgvector_matches_or_beats_faiss(retriever, pgvector_retriever):
    faiss = _score(retriever)
    pgvector = _score(pgvector_retriever)
    n = len(RETRIEVAL_CASES)
    print(f"\n{'metric':<16}{'faiss':>10}{'pgvector':>10}   (n={n})")
    for metric in faiss:
        fmt = "{:>10.3f}" if metric == "mrr@5" else "{:>10d}"
        print(f"{metric:<16}" + fmt.format(faiss[metric]) + fmt.format(pgvector[metric]))

    assert pgvector["category_hit@5"] >= faiss["category_hit@5"]
    assert pgvector["keyword_hit@5"] >= faiss["keyword_hit@5"]
    assert pgvector["mrr@5"] >= faiss["mrr@5"] - 1e-9


def test_category_filter_is_exact(pgvector_retriever):
    results = pgvector_retriever.search("tax-loss harvesting", top_k=5, category_filter="tax_accounts")
    assert results and all(r["category"] == "tax_accounts" for r in results)
