"""Phase 4: knowledge base and FAQ cache on Postgres + pgvector.

Pure logic (tsquery building, rank fusion, backend selection) always runs.
Everything touching the database runs when FINNIE_TEST_DATABASE_URL points at
a disposable Postgres with pgvector available — these tests TRUNCATE kb_chunks
and faq_cache:

    FINNIE_TEST_DATABASE_URL=postgresql://user@localhost:5432/finnie_test pytest tests/test_rag_pgvector.py

A deterministic fake embedder keeps them fast and exact. Retrieval *quality*
against FAISS, with the real model and knowledge base, is measured separately
in tests/evals/test_rag_backend_parity.py.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.rag.pgvector import keyword_query, reciprocal_rank_fusion, vector_literal

_PG_URL = os.environ.get("FINNIE_TEST_DATABASE_URL", "")
needs_pg = pytest.mark.skipif(not _PG_URL, reason="set FINNIE_TEST_DATABASE_URL")


class FakeEmbedder:
    """Bag-of-words hashed into 384 dims; records every call.

    md5, not hash(): Python string hashing is randomised per process.
    `blind_to` words are ignored, to model semantic search missing a term.
    """

    def __init__(self, blind_to: tuple[str, ...] = ()) -> None:
        self.calls: list[list[str]] = []
        self._blind = set(blind_to)

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word not in self._blind:
                    out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    @property
    def embedded_texts(self) -> int:
        return sum(len(call) for call in self.calls)


# ── Pure logic ─────────────────────────────────────────────────────────────

class TestKeywordQuery:
    def test_or_joins_distinct_words(self):
        assert keyword_query("What is a P/E ratio? Ratio!") == "what | is | ratio"

    def test_cannot_inject_tsquery_operators(self):
        query = keyword_query("bonds' & !stocks | (etf):* <-> 'x")
        assert set(query) <= set("abcdefghijklmnopqrstuvwxyz0123456789 |")
        assert query == "bonds | stocks | etf"

    def test_keeps_numbers(self):
        assert keyword_query("the rule of 110 and 401k") == "the | rule | of | 110 | and | 401k"

    def test_empty(self):
        assert keyword_query("?! a") == ""


class TestRankFusion:
    def test_agreement_beats_one_strong_ranker(self):
        fused = [cid for cid, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)]
        assert fused[0] == "b"          # rank 2 + rank 1 beats rank 1 + rank 3

    def test_single_list_keeps_order_and_scores(self):
        fused = reciprocal_rank_fusion([["x", "y"]], k=60)
        assert [cid for cid, _ in fused] == ["x", "y"]
        assert fused[0][1] == pytest.approx(1 / 61)

    def test_items_only_in_one_list_are_kept(self):
        assert {cid for cid, _ in reciprocal_rank_fusion([["a"], ["b"]])} == {"a", "b"}


def test_vector_literal():
    assert vector_literal(np.array([0.5, -1.0, 1e-9], dtype=np.float32)) == "[0.5,-1,1e-09]"


class TestBackendSelection:
    @pytest.mark.parametrize("backend,url,expected", [
        ("auto", "", "faiss"),
        ("auto", "postgresql://x", "pgvector"),
        ("faiss", "postgresql://x", "faiss"),
        ("pgvector", "postgresql://x", "pgvector"),
    ])
    def test_rag_backend(self, monkeypatch, backend, url, expected):
        from src.core.config import get_settings
        from src.rag.retriever import rag_backend_name

        settings = get_settings()
        monkeypatch.setattr(settings.rag, "backend", backend)
        monkeypatch.setattr(settings, "database_url", url)
        assert rag_backend_name() == expected

    def test_pgvector_without_database_is_an_error(self, monkeypatch):
        from src.core.config import get_settings
        from src.rag.retriever import rag_backend_name

        monkeypatch.setattr(get_settings().rag, "backend", "pgvector")
        with pytest.raises(RuntimeError):
            rag_backend_name()

    def test_default_suite_uses_faiss_and_redis_cache(self):
        from src.rag.retriever import RAGRetriever, get_retriever
        from src.utils.semantic_cache import SemanticFAQCache, get_faq_cache

        assert isinstance(get_retriever(), RAGRetriever)
        assert isinstance(get_faq_cache(), SemanticFAQCache)


def test_context_format_is_shared_by_both_backends():
    from src.rag.retriever import format_context

    chunks = [{"title": "T1", "source": "a/b.txt", "text": "one"}, {"title": "T2", "source": "c.txt", "text": "two"}]
    assert format_context(chunks) == "[Source 1: T1 (a/b.txt)]\none\n\n---\n\n[Source 2: T2 (c.txt)]\ntwo"
    assert format_context([]) == ""


# ── Database ───────────────────────────────────────────────────────────────

@pytest.fixture
def pg(monkeypatch):
    """Real Postgres: DATABASE_URL set, schema created, tables emptied."""
    from src.core.config import get_settings
    from src.persistence.backend import get_pool, reset_persistence
    from src.rag.pgvector import ensure_schema
    from src.rag.retriever import get_retriever
    from src.utils.semantic_cache import get_faq_cache

    monkeypatch.setenv("DATABASE_URL", _PG_URL)
    for cached in (get_settings, ensure_schema, get_retriever, get_faq_cache):
        cached.cache_clear()
    reset_persistence()
    ensure_schema()
    with get_pool().connection() as conn:
        conn.execute("TRUNCATE kb_chunks, faq_cache")
    yield get_pool()
    reset_persistence()
    for cached in (get_settings, ensure_schema, get_retriever, get_faq_cache):
        cached.cache_clear()


def _article(root: Path, category: str, name: str, title: str, body: str) -> Path:
    path = root / category / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{title}\n\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture
def kb(tmp_path) -> Path:
    root = tmp_path / "kb"
    _article(root, "investing_basics", "bonds", "Bonds Explained",
             "A bond is a loan to a government or company that pays a fixed coupon of interest "
             "until maturity, when the principal is repaid to the bondholder.")
    _article(root, "investing_basics", "stocks", "Stocks Explained",
             "A stock is a share of ownership in a company; shareholders may receive dividends and "
             "benefit when the share price rises, but they also carry the risk of loss.")
    _article(root, "tax_accounts", "harvesting", "Tax Loss Harvesting",
             "Tax loss harvesting sells an investment at a loss to offset capital gains, while the "
             "wash sale rule forbids buying it back within thirty days of the sale.")
    return root


def _count(pool, where: str = "TRUE") -> int:
    with pool.connection() as conn:
        return conn.execute(f"SELECT count(*) AS n FROM kb_chunks WHERE {where}").fetchone()["n"]


@needs_pg
class TestSchema:
    def test_tables_and_extension_exist(self, pg):
        with pg.connection() as conn:
            tables = {r["t"] for r in conn.execute(
                "SELECT tablename AS t FROM pg_tables WHERE tablename IN ('kb_chunks', 'faq_cache')")}
            assert tables == {"kb_chunks", "faq_cache"}
            assert conn.execute("SELECT 1 AS ok FROM pg_extension WHERE extname = 'vector'").fetchone()

    def test_dimension_change_is_caught_not_silently_broken(self, pg, monkeypatch):
        from src.core.config import get_settings
        from src.rag.pgvector import ensure_schema

        ensure_schema.cache_clear()
        monkeypatch.setattr(get_settings().embeddings, "dimension", 128)
        with pytest.raises(RuntimeError, match="migrate"):
            ensure_schema()


@needs_pg
class TestSync:
    def test_first_sync_inserts_every_chunk(self, pg, kb):
        from src.rag.pgvector import sync_knowledge_base

        report = sync_knowledge_base(kb, FakeEmbedder())
        assert (report.inserted, report.updated, report.deleted) == (3, 0, 0)
        assert _count(pg) == 3

    def test_unchanged_resync_embeds_nothing(self, pg, kb):
        from src.rag.pgvector import sync_knowledge_base

        sync_knowledge_base(kb, FakeEmbedder())
        embedder = FakeEmbedder()
        report = sync_knowledge_base(kb, embedder)
        assert report.unchanged == 3 and report.inserted == report.updated == report.deleted == 0
        assert embedder.calls == []

    def test_edit_reembeds_only_that_chunk_and_is_searchable(self, pg, kb):
        from src.rag.pgvector import PgVectorRetriever, sync_knowledge_base

        sync_knowledge_base(kb, FakeEmbedder())
        _article(kb, "investing_basics", "bonds", "Bonds Explained",
                 "A bond is a loan that pays a coupon; a zero coupon treasury strip instead sells "
                 "at a deep discount and repays full face value at maturity to the investor.")
        embedder = FakeEmbedder()
        report = sync_knowledge_base(kb, embedder)

        assert (report.updated, report.unchanged) == (1, 2)
        assert embedder.embedded_texts == 1
        hits = PgVectorRetriever(embed_fn=embedder).search("treasury strip discount", top_k=1)
        assert hits[0]["source"] == "investing_basics/bonds.txt"

    def test_deleted_file_removes_its_chunks(self, pg, kb):
        from src.rag.pgvector import sync_knowledge_base

        sync_knowledge_base(kb, FakeEmbedder())
        (kb / "investing_basics" / "stocks.txt").unlink()
        report = sync_knowledge_base(kb, FakeEmbedder())
        assert report.deleted == 1 and _count(pg) == 2

    def test_new_embedding_model_reembeds_everything(self, pg, kb, monkeypatch):
        from src.core.config import get_settings
        from src.rag.pgvector import sync_knowledge_base

        sync_knowledge_base(kb, FakeEmbedder())
        monkeypatch.setattr(get_settings().embeddings, "model", "another-model")
        embedder = FakeEmbedder()
        report = sync_knowledge_base(kb, embedder)
        assert report.updated == 3 and embedder.embedded_texts == 3

    def test_rows_from_other_origins_survive_a_sync(self, pg, kb):
        """Curated rows imported from elsewhere must not be deleted by a repo sync."""
        from src.rag.pgvector import sync_knowledge_base, vector_literal

        with pg.connection() as conn:
            conn.execute(
                "INSERT INTO kb_chunks (chunk_id, content_hash, category, source, title, text, "
                "embedding, embedding_model, origin) VALUES ('faq_1', 'h', 'faq', 'meko', 'FAQ', "
                "'curated answer text', %s::vector, 'm', 'meko_promoted')",
                (vector_literal(np.ones(384) / np.sqrt(384)),),
            )
        sync_knowledge_base(kb, FakeEmbedder())
        assert _count(pg, "origin = 'meko_promoted'") == 1

    def test_concurrent_syncs_serialise(self, pg, kb):
        """Instances starting together must not race on the upsert."""
        from src.rag.pgvector import sync_knowledge_base

        errors: list[BaseException] = []

        def run() -> None:
            try:
                sync_knowledge_base(kb, FakeEmbedder())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [] and _count(pg) == 3


@needs_pg
class TestHybridSearch:
    @pytest.fixture
    def retriever(self, pg, kb):
        from src.rag.pgvector import PgVectorRetriever, sync_knowledge_base

        embedder = FakeEmbedder()
        sync_knowledge_base(kb, embedder)
        return PgVectorRetriever(embed_fn=embedder)

    def test_result_shape_and_top_k(self, retriever):
        hits = retriever.search("what is a bond coupon", top_k=2)
        assert len(hits) == 2
        assert {"chunk_id", "source", "category", "title", "text", "score", "similarity"} <= set(hits[0])
        assert hits[0]["source"] == "investing_basics/bonds.txt"

    def test_category_filter_is_exact(self, retriever):
        hits = retriever.search("bond stock tax loss", top_k=5, category_filter="tax_accounts")
        assert hits and {h["category"] for h in hits} == {"tax_accounts"}

    def test_keyword_ranker_rescues_a_term_semantics_misses(self, pg, kb):
        """With an embedder blind to 'wash', only the keyword ranker can find the
        wash-sale chunk — hybrid search must still put it first."""
        from src.rag.pgvector import PgVectorRetriever, sync_knowledge_base

        embedder = FakeEmbedder(blind_to=("wash",))
        sync_knowledge_base(kb, embedder)
        hits = PgVectorRetriever(embed_fn=embedder).search("wash", top_k=1)
        assert hits[0]["source"] == "tax_accounts/harvesting.txt"

    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.search("   ") == []

    def test_get_context_uses_the_shared_format(self, retriever):
        context = retriever.get_context("bond coupon", top_k=1)
        # Titles come from the file name, as with the FAISS index.
        assert context.startswith("[Source 1: Bonds (investing_basics/bonds.txt)]\n")

    def test_backend_auto_selects_pgvector_with_a_database(self, pg):
        from src.rag.pgvector import PgVectorRetriever
        from src.rag.retriever import get_retriever

        assert isinstance(get_retriever(), PgVectorRetriever)


# ── FAQ cache on Postgres ──────────────────────────────────────────────────

_VECTORS = {
    "what is a p e ratio": [1.0, 0.0, 0.0],
    "whats the p e ratio": [0.995, 0.1, 0.0],       # ~0.995 cosine -> hit
    "explain price to earnings": [0.8, 0.6, 0.0],   # 0.80 -> below threshold
}


def _faq_embed(calls: list):
    from src.utils.semantic_cache import normalize_query

    def embed(text: str) -> np.ndarray:
        calls.append(text)
        vector = np.zeros(384, dtype=np.float32)
        vector[:3] = _VECTORS.get(normalize_query(text), [0.0, 0.0, 1.0])
        return vector

    return embed


@needs_pg
class TestPostgresFAQCache:
    @pytest.fixture
    def cache(self, pg):
        from src.utils.semantic_cache import PostgresFAQCache

        self.calls: list[str] = []
        return PostgresFAQCache(embed_fn=_faq_embed(self.calls))

    def test_exact_hit_needs_no_embedding(self, cache):
        cache.set("What is a P/E ratio?", "Price over earnings.", "finance_qa")
        self.calls.clear()
        hit = cache.get("what is a p/e ratio")
        assert hit["match_type"] == "exact" and hit["answer"] == "Price over earnings."
        assert self.calls == []

    def test_semantic_hit_and_threshold(self, cache):
        cache.set("What is a P/E ratio?", "Price over earnings.", "finance_qa")
        hit = cache.get("What's the P/E ratio?")
        assert hit["match_type"] == "semantic" and hit["similarity"] >= 0.92
        assert cache.get("Explain price to earnings") is None

    def test_shared_between_instances(self, pg, cache):
        """The point of moving it to Postgres: one instance's write serves another."""
        from src.utils.semantic_cache import PostgresFAQCache

        cache.set("What is a P/E ratio?", "Price over earnings.", "finance_qa")
        other = PostgresFAQCache(embed_fn=_faq_embed([]))
        assert other.get("What is a P/E ratio?")["answer"] == "Price over earnings."

    def test_knowledge_base_or_tax_year_change_invalidates(self, cache):
        from src.utils.semantic_cache import current_tax_year

        cache.set("What is a P/E ratio?", "old answer", "finance_qa")
        with patch("src.utils.semantic_cache.get_kb_version", return_value="changed"):
            assert cache.get("What is a P/E ratio?") is None
        with patch("src.utils.semantic_cache.current_tax_year", return_value=current_tax_year() + 1):
            assert cache.get("What is a P/E ratio?") is None

    def test_expired_entries_are_not_served(self, pg, cache):
        cache.set("What is a P/E ratio?", "answer", "finance_qa")
        with pg.connection() as conn:
            conn.execute("UPDATE faq_cache SET expires_at = now() - interval '1 second'")
        assert cache.get("What is a P/E ratio?") is None

    def test_capped_at_max_entries_keeping_newest(self, pg, cache, monkeypatch):
        monkeypatch.setattr(cache._config, "max_entries", 3)
        for i in range(5):
            cache.set(f"question number {i}", f"answer {i}", "finance_qa")
        with pg.connection() as conn:
            kept = {r["query"] for r in conn.execute("SELECT query FROM faq_cache")}
        assert kept == {"question number 2", "question number 3", "question number 4"}

    def test_database_errors_fail_open(self, cache, monkeypatch):
        def broken_pool():
            raise RuntimeError("database down")

        monkeypatch.setattr(cache, "_pool", broken_pool)
        assert cache.get("What is a P/E ratio?") is None
        assert cache.set("q", "a", "finance_qa") is False

    def test_backend_auto_selects_postgres_with_a_database(self, pg):
        from src.utils.semantic_cache import PostgresFAQCache, get_faq_cache

        assert isinstance(get_faq_cache(), PostgresFAQCache)
