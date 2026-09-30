"""Semantic answer cache for repeat FAQ-style questions.

A hit skips the classifier and the agent entirely, so a turn costs zero LLM
calls. Two tiers:

  1. exact — SHA-256 of the normalised query, a single Redis lookup;
  2. semantic — cosine over cached query embeddings, served at or above
     ``faq_cache.similarity_threshold``.

Correctness rules (a wrong figure is worse than a miss, so the cache is
deliberately conservative):

  * only answers that used no live market/macro data are ever written
    (enforced by the caller, ``src/workflow/faq_cache.py``);
  * every entry is stamped with the knowledge-base version and the tax year it
    was produced under, and is ignored once either moves.

Every operation fails open: any backend or embedding error logs and behaves as
a miss, so the normal workflow still answers.
"""
from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, Optional

import numpy as np

from src.core.config import get_settings
from src.rag.version import get_kb_version
from src.utils.cache import get_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_KEY_PARTS = ("faq", "index")
_ENTRY_KEY_PARTS = ("faq", "entry")
_REFRESH_INTERVAL_S = 60.0

_APOSTROPHE_RE = re.compile(r"['\u2019]")
_PUNCT_RE = re.compile(r"[^\w\s%$]")
_WS_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace for exact matching.

    Apostrophes are deleted rather than replaced with a space, so "what's" and
    "whats" normalise alike instead of becoming "what s".
    """
    lowered = _APOSTROPHE_RE.sub("", query.lower())
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def query_hash(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()[:32]


def current_tax_year() -> int:
    return datetime.now(timezone.utc).year


class SemanticFAQCache:
    """Redis-backed exact + semantic cache over prior answers."""

    def __init__(self, embed_fn: Optional[Callable[[str], np.ndarray]] = None) -> None:
        self._settings = get_settings()
        self._config = self._settings.fast_path.faq_cache
        self._cache = get_cache()
        self._embed_fn = embed_fn
        self._matrix: Optional[np.ndarray] = None
        self._entries: list[dict[str, Any]] = []
        self._loaded_at: float = 0.0

    # ── embedding ────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is not None:
            vector = np.asarray(self._embed_fn(text), dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            return vector / norm if norm else vector
        from src.core.embeddings import embed_one

        return embed_one(text)

    # ── keys ─────────────────────────────────────────────────────────────────

    def _index_key(self) -> str:
        return self._cache.cache_key(*_INDEX_KEY_PARTS)

    def _entry_key(self, digest: str) -> str:
        return self._cache.cache_key(*_ENTRY_KEY_PARTS, digest)

    # ── validity ─────────────────────────────────────────────────────────────

    def _is_valid(self, entry: dict[str, Any]) -> bool:
        """An entry is servable only under the KB version and tax year it was written for."""
        return (
            entry.get("kb_version") == get_kb_version()
            and entry.get("tax_year") == current_tax_year()
        )

    # ── in-memory index ──────────────────────────────────────────────────────

    def _refresh(self, force: bool = False) -> None:
        """Reload entries from the shared cache into the local vector matrix."""
        if not force and (time.monotonic() - self._loaded_at) < _REFRESH_INTERVAL_S:
            return

        digests = self._cache.get(self._index_key()) or []
        entries: list[dict[str, Any]] = []
        for digest in digests:
            entry = self._cache.get(self._entry_key(digest))
            if entry and entry.get("embedding"):
                entries.append(entry)

        self._entries = entries
        self._matrix = (
            np.asarray([e["embedding"] for e in entries], dtype=np.float32)
            if entries
            else None
        )
        self._loaded_at = time.monotonic()

    # ── read ─────────────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[dict[str, Any]]:
        """Return a cached answer for `query`, or None on any miss or error."""
        if not self._config.enabled or not query.strip():
            return None

        try:
            # Tier 1 — exact match on the normalised query.
            entry = self._cache.get(self._entry_key(query_hash(query)))
            if entry and self._is_valid(entry):
                return {**entry, "match_type": "exact", "similarity": 1.0}

            # Tier 2 — nearest cached query by cosine.
            self._refresh()
            if self._matrix is None or not len(self._matrix):
                return None

            similarities = self._matrix @ self._embed(query)
            best = int(np.argmax(similarities))
            score = float(similarities[best])
            if score < self._config.similarity_threshold:
                return None

            candidate = self._entries[best]
            if not self._is_valid(candidate):
                return None
            return {**candidate, "match_type": "semantic", "similarity": score}

        except Exception as exc:  # noqa: BLE001 — a cache error must never fail the turn
            logger.warning("faq_cache_get_error", error=str(exc))
            return None

    # ── write ────────────────────────────────────────────────────────────────

    def set(self, query: str, answer: str, agent: str) -> bool:
        """Store an answer. Returns True when written."""
        if not self._config.enabled or not query.strip() or not answer.strip():
            return False

        try:
            digest = query_hash(query)
            entry = {
                "query": query,
                "query_norm": normalize_query(query),
                "answer": answer,
                "agent": agent,
                "kb_version": get_kb_version(),
                "tax_year": current_tax_year(),
                "embedding": self._embed(query).tolist(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._cache.set(self._entry_key(digest), entry, ttl=self._config.ttl_seconds)

            digests = self._cache.get(self._index_key()) or []
            if digest not in digests:
                digests.append(digest)
                # Bound the index; oldest entries age out of Redis by TTL anyway.
                if len(digests) > self._config.max_entries:
                    digests = digests[-self._config.max_entries :]
                self._cache.set(self._index_key(), digests, ttl=self._config.ttl_seconds)

            self._refresh(force=True)
            logger.info("faq_cache_write", agent=agent, entries=len(self._entries))
            return True

        except Exception as exc:  # noqa: BLE001
            logger.warning("faq_cache_set_error", error=str(exc))
            return False

    def clear(self) -> None:
        """Drop every entry (used by tests and by the nightly purge)."""
        for digest in self._cache.get(self._index_key()) or []:
            self._cache.delete(self._entry_key(digest))
        self._cache.delete(self._index_key())
        self._entries = []
        self._matrix = None
        self._loaded_at = 0.0




class PostgresFAQCache:
    """The FAQ cache in Postgres (faq_cache table), shared by every app instance.

    Same interface, validity rules and fail-open contract as SemanticFAQCache.
    An exact match is answered without embedding the question; only a miss
    pays for the embedding and the similarity search. Validity (knowledge-base
    version, tax year, expiry) is enforced in the WHERE clause, so a stale
    entry can never be returned even before the purge runs.
    """

    def __init__(self, embed_fn: Optional[Callable[[str], np.ndarray]] = None) -> None:
        self._config = get_settings().fast_path.faq_cache
        self._embed_fn = embed_fn

    def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is not None:
            vector = np.asarray(self._embed_fn(text), dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            return vector / norm if norm else vector
        from src.core.embeddings import embed_one

        return embed_one(text)

    @staticmethod
    def _pool():
        from src.rag.pgvector import ensure_schema
        from src.persistence.backend import get_pool

        ensure_schema()
        return get_pool()

    def _validity(self) -> dict[str, Any]:
        return {"kv": get_kb_version(), "ty": current_tax_year()}

    def get(self, query: str) -> Optional[dict[str, Any]]:
        """Return a cached answer for `query`, or None on any miss or error."""
        if not self._config.enabled or not query.strip():
            return None
        try:
            from src.rag.pgvector import vector_literal

            valid = self._validity()
            with self._pool().connection() as conn:
                row = conn.execute(
                    "SELECT answer, agent FROM faq_cache WHERE query_hash = %(h)s "
                    "AND kb_version = %(kv)s AND tax_year = %(ty)s AND expires_at > now()",
                    {"h": query_hash(query), **valid},
                ).fetchone()
                if row:
                    return {**row, "match_type": "exact", "similarity": 1.0}

                row = conn.execute(
                    "SELECT answer, agent, 1 - (embedding <=> %(q)s::vector) AS similarity "
                    "FROM faq_cache WHERE kb_version = %(kv)s AND tax_year = %(ty)s "
                    "AND expires_at > now() ORDER BY embedding <=> %(q)s::vector LIMIT 1",
                    {"q": vector_literal(self._embed(query)), **valid},
                ).fetchone()
            if not row or float(row["similarity"]) < self._config.similarity_threshold:
                return None
            return {
                "answer": row["answer"],
                "agent": row["agent"],
                "match_type": "semantic",
                "similarity": float(row["similarity"]),
            }
        except Exception as exc:  # noqa: BLE001 — a cache error must never fail the turn
            logger.warning("faq_cache_get_error", error=str(exc), backend="postgres")
            return None

    def set(self, query: str, answer: str, agent: str) -> bool:
        """Store an answer, purge expired rows, and cap the table at max_entries."""
        if not self._config.enabled or not query.strip() or not answer.strip():
            return False
        try:
            from src.rag.pgvector import vector_literal

            with self._pool().connection() as conn, conn.transaction():
                conn.execute(
                    """
                    INSERT INTO faq_cache (query_hash, query, answer, agent, embedding,
                                           kb_version, tax_year, created_at, expires_at)
                    VALUES (%(h)s, %(query)s, %(answer)s, %(agent)s, %(q)s::vector, %(kv)s, %(ty)s,
                            now(), now() + make_interval(secs => %(ttl)s))
                    ON CONFLICT (query_hash) DO UPDATE SET
                        query = EXCLUDED.query, answer = EXCLUDED.answer, agent = EXCLUDED.agent,
                        embedding = EXCLUDED.embedding, kb_version = EXCLUDED.kb_version,
                        tax_year = EXCLUDED.tax_year, created_at = now(), expires_at = EXCLUDED.expires_at
                    """,
                    {
                        "h": query_hash(query), "query": query, "answer": answer, "agent": agent,
                        "q": vector_literal(self._embed(query)), "ttl": self._config.ttl_seconds,
                        **self._validity(),
                    },
                )
                conn.execute("DELETE FROM faq_cache WHERE expires_at <= now()")
                conn.execute(
                    "DELETE FROM faq_cache WHERE query_hash IN (SELECT query_hash FROM faq_cache "
                    "ORDER BY created_at DESC OFFSET %s)",
                    (self._config.max_entries,),
                )
            logger.info("faq_cache_write", agent=agent, backend="postgres")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("faq_cache_set_error", error=str(exc), backend="postgres")
            return False

    def clear(self) -> None:
        with self._pool().connection() as conn:
            conn.execute("DELETE FROM faq_cache")


def faq_cache_backend() -> str:
    """"postgres" or "redis", resolving the "auto" setting."""
    settings = get_settings()
    backend = settings.fast_path.faq_cache.backend
    if backend == "auto":
        return "postgres" if settings.database_url else "redis"
    if backend == "postgres" and not settings.database_url:
        raise RuntimeError("fast_path.faq_cache.backend is 'postgres' but DATABASE_URL is not set")
    return backend


@lru_cache(maxsize=1)
def get_faq_cache() -> Any:
    """The process-wide FAQ cache: Postgres (shared) or Redis/in-process."""
    if faq_cache_backend() == "postgres":
        return PostgresFAQCache()
    return SemanticFAQCache()
