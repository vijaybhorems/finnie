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


@lru_cache(maxsize=1)
def get_faq_cache() -> SemanticFAQCache:
    return SemanticFAQCache()
