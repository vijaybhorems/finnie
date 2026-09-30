"""Knowledge base on Postgres + pgvector: schema, sync, and hybrid search.

Tables (created idempotently by ensure_schema):

  kb_chunks   one row per knowledge-base chunk: text, a generated tsvector for
              keyword search, the embedding, and a content hash that lets sync
              skip unchanged chunks. `origin` separates chunks synced from the
              repo ('repo') from ones imported elsewhere (e.g. curated FAQs),
              so a repo sync never deletes the latter.
  faq_cache   the semantic answer cache (src/utils/semantic_cache.py), shared
              by every app instance instead of living in one process's memory.

Search is exact cosine, not an ANN index. At this corpus size (tens to low
thousands of chunks) a sequential scan is sub-millisecond, has perfect recall,
and has none of HNSW's filtered-search pitfall: an HNSW scan with a category
WHERE can return fewer than k rows. It also needs no vendor-specific index
type (YugabyteDB uses ybhnsw). Add an HNSW index once kb_chunks passes roughly
10k rows and measure recall when you do.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.core.config import get_settings
from src.rag.indexer import chunk_documents, load_documents
from src.utils.logger import get_logger

logger = get_logger(__name__)

# pg_advisory_xact_lock key serialising schema creation and sync across app
# instances starting at once ("FINNIE" as an integer).
_LOCK_KEY = 0x46494E4E4945

EmbedFn = Callable[[list[str]], np.ndarray]


def _default_embed(texts: list[str]) -> np.ndarray:
    from src.core.embeddings import embed

    return embed(texts)


def vector_literal(vector: Any) -> str:
    """pgvector text form ("[0.1,0.2,...]") — no client-side type adapter needed."""
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


_TOKEN = re.compile(r"[a-z0-9]+")


def keyword_query(text: str) -> str:
    """An OR tsquery over the question's words, safe to pass to to_tsquery.

    plainto_tsquery/websearch_to_tsquery AND every term, so a natural question
    ("What is a P/E ratio and why does it matter?") would match almost nothing.
    Tokens are restricted to [a-z0-9] so no tsquery operator can be injected;
    one-character tokens are dropped as noise, and to_tsquery removes English
    stopwords itself.
    """
    seen: dict[str, None] = {}
    for token in _TOKEN.findall(text.lower()):
        if len(token) > 1:
            seen.setdefault(token, None)
    return " | ".join(seen)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank), rank from 1.

    Rank-based, so it needs no calibration between cosine similarity and
    ts_rank, whose scales are unrelated. Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _pool():
    from src.persistence.backend import get_pool

    return get_pool()


@lru_cache(maxsize=1)
def ensure_schema() -> None:
    """Create the pgvector extension, tables and indexes if missing (idempotent)."""
    from psycopg import sql

    dimension = get_settings().embeddings.dimension
    with _pool().connection() as conn, conn.transaction():
        # Concurrent CREATE ... IF NOT EXISTS can still collide on the catalog.
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as exc:
            raise RuntimeError(
                "pgvector is not available: run CREATE EXTENSION vector as a user "
                "allowed to (on Cloud SQL, enable the extension; on Neon it is built in)"
            ) from exc

        conn.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                chunk_id        text PRIMARY KEY,
                content_hash    text NOT NULL,
                category        text NOT NULL,
                source          text NOT NULL,
                title           text NOT NULL,
                text            text NOT NULL,
                tsv             tsvector GENERATED ALWAYS AS
                                  (to_tsvector('english', title || ' ' || text)) STORED,
                embedding       vector({dim}) NOT NULL,
                embedding_model text NOT NULL,
                origin          text NOT NULL DEFAULT 'repo',
                updated_at      timestamptz NOT NULL DEFAULT now()
            )""").format(dim=sql.Literal(dimension)))
        conn.execute("CREATE INDEX IF NOT EXISTS kb_chunks_tsv_idx ON kb_chunks USING gin (tsv)")
        conn.execute("CREATE INDEX IF NOT EXISTS kb_chunks_category_idx ON kb_chunks (category)")

        conn.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS faq_cache (
                query_hash  text PRIMARY KEY,
                query       text NOT NULL,
                answer      text NOT NULL,
                agent       text NOT NULL,
                embedding   vector({dim}) NOT NULL,
                kb_version  text NOT NULL,
                tax_year    int NOT NULL,
                created_at  timestamptz NOT NULL DEFAULT now(),
                expires_at  timestamptz NOT NULL
            )""").format(dim=sql.Literal(dimension)))
        conn.execute("CREATE INDEX IF NOT EXISTS faq_cache_expires_idx ON faq_cache (expires_at)")

        # A changed embedding dimension needs a migration, not a silent failure
        # on the first insert.
        for table in ("kb_chunks", "faq_cache"):
            column_type = conn.execute(
                "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (table,),
            ).fetchone()["t"]
            if column_type != f"vector({dimension})":
                raise RuntimeError(
                    f"{table}.embedding is {column_type} but embeddings.dimension is "
                    f"{dimension}: the embedding model changed dimension — migrate the table"
                )
    logger.info("pgvector_schema_ready", dimension=dimension)


# ── Sync ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SyncReport:
    inserted: int
    updated: int
    deleted: int
    unchanged: int


def _content_hash(chunk: dict[str, Any]) -> str:
    payload = "\x1f".join((chunk["category"], chunk["source"], chunk["title"], chunk["text"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sync_knowledge_base(
    kb_path: Optional[Path] = None,
    embed_fn: Optional[EmbedFn] = None,
) -> SyncReport:
    """Make kb_chunks match the knowledge-base files.

    Idempotent and cheap when nothing changed: only new or edited chunks (or
    ones embedded by a different model) are embedded; chunks whose files are
    gone are deleted. Rows not from the repo are never touched. Serialised by
    an advisory lock, so instances starting together don't race.
    """
    settings = get_settings()
    kb_path = Path(kb_path) if kb_path else settings.knowledge_base_path
    embed_fn = embed_fn or _default_embed
    model = settings.embeddings.model

    chunks = chunk_documents(load_documents(kb_path), settings.rag.chunk_size, settings.rag.chunk_overlap)
    wanted = {chunk["chunk_id"]: {**chunk, "content_hash": _content_hash(chunk)} for chunk in chunks}

    ensure_schema()
    with _pool().connection() as conn, conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
        existing = {
            row["chunk_id"]: row
            for row in conn.execute(
                "SELECT chunk_id, content_hash, embedding_model FROM kb_chunks WHERE origin = 'repo'"
            ).fetchall()
        }

        stale_ids = [cid for cid in existing if cid not in wanted]
        changed_ids = [
            cid for cid, chunk in wanted.items()
            if cid not in existing
            or existing[cid]["content_hash"] != chunk["content_hash"]
            or existing[cid]["embedding_model"] != model
        ]

        if changed_ids:
            vectors = embed_fn([wanted[cid]["text"] for cid in changed_ids])
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO kb_chunks (chunk_id, content_hash, category, source, title, text,
                                           embedding, embedding_model, origin, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, 'repo', now())
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash, category = EXCLUDED.category,
                        source = EXCLUDED.source, title = EXCLUDED.title, text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model,
                        origin = 'repo', updated_at = now()
                    """,
                    [
                        (cid, wanted[cid]["content_hash"], wanted[cid]["category"], wanted[cid]["source"],
                         wanted[cid]["title"], wanted[cid]["text"], vector_literal(vector), model)
                        for cid, vector in zip(changed_ids, vectors)
                    ],
                )
        if stale_ids:
            conn.execute(
                "DELETE FROM kb_chunks WHERE origin = 'repo' AND chunk_id = ANY(%s)", (stale_ids,)
            )

    inserted = sum(1 for cid in changed_ids if cid not in existing)
    report = SyncReport(
        inserted=inserted,
        updated=len(changed_ids) - inserted,
        deleted=len(stale_ids),
        unchanged=len(wanted) - len(changed_ids),
    )
    logger.info("kb_sync_complete", **report.__dict__)
    return report


# ── Search ───────────────────────────────────────────────────────────────────

_HYBRID_SQL = """
WITH semantic AS (
    SELECT chunk_id,
           row_number() OVER (ORDER BY embedding <=> %(q)s::vector) AS rank,
           1 - (embedding <=> %(q)s::vector) AS similarity
    FROM kb_chunks
    WHERE embedding_model = %(model)s AND (%(category)s::text IS NULL OR category = %(category)s)
    ORDER BY embedding <=> %(q)s::vector
    LIMIT %(n)s
),
keyword AS (
    SELECT chunk_id,
           row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC, chunk_id) AS rank
    FROM kb_chunks, to_tsquery('english', %(tsq)s) AS query
    WHERE tsv @@ query AND (%(category)s::text IS NULL OR category = %(category)s)
    ORDER BY ts_rank_cd(tsv, query) DESC, chunk_id
    LIMIT %(n)s
)
SELECT c.chunk_id, c.source, c.category, c.title, c.text,
       semantic.rank AS semantic_rank, semantic.similarity, keyword.rank AS keyword_rank
FROM kb_chunks c
LEFT JOIN semantic USING (chunk_id)
LEFT JOIN keyword USING (chunk_id)
WHERE semantic.chunk_id IS NOT NULL OR keyword.chunk_id IS NOT NULL
"""


class PgVectorRetriever:
    """Hybrid (semantic + keyword) search over kb_chunks, fused by RRF.

    Same interface and result shape as the FAISS RAGRetriever; `score` is the
    fused RRF score (not a cosine similarity), and `similarity` carries the
    cosine where the semantic ranker returned the chunk.
    """

    def __init__(self, embed_fn: Optional[EmbedFn] = None) -> None:
        self._settings = get_settings()
        self._top_k = self._settings.rag.top_k
        self._embed_fn = embed_fn or _default_embed

    def warm_up(self) -> bool:
        """Create the schema, sync the knowledge base, and prime the embedder."""
        ensure_schema()
        sync_knowledge_base(embed_fn=self._embed_fn)
        self._embed_fn(["warm up"])  # first torch inference is slow; pay it now
        return True

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self._top_k
        if not query.strip():
            return []
        ensure_schema()
        params = {
            "q": vector_literal(self._embed_fn([query])[0]),
            "model": self._settings.embeddings.model,
            "category": category_filter,
            "tsq": keyword_query(query),
            "n": max(self._settings.rag.hybrid_candidates, k),
        }
        with _pool().connection() as conn:
            rows = {row["chunk_id"]: row for row in conn.execute(_HYBRID_SQL, params).fetchall()}

        def ranked_by(column: str) -> list[str]:
            hits = [row for row in rows.values() if row[column] is not None]
            return [row["chunk_id"] for row in sorted(hits, key=lambda row: row[column])]

        fused = reciprocal_rank_fusion(
            [ranked_by("semantic_rank"), ranked_by("keyword_rank")], k=self._settings.rag.rrf_k
        )[:k]
        return [
            {
                "chunk_id": cid,
                "source": rows[cid]["source"],
                "category": rows[cid]["category"],
                "title": rows[cid]["title"],
                "text": rows[cid]["text"],
                "score": score,
                "similarity": float(rows[cid]["similarity"]) if rows[cid]["similarity"] is not None else None,
            }
            for cid, score in fused
        ]

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> str:
        from src.rag.retriever import format_context

        return format_context(self.search(query, top_k=top_k, category_filter=category_filter))
