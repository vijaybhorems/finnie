"""RAG retriever — semantic search over the FAISS index."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from src.core.config import get_settings
from src.rag.indexer import RAGIndexer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """Loads FAISS index and provides semantic search."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._index_path = self._settings.rag_index_path
        self._top_k = self._settings.rag.top_k
        self._index = None
        self._metadata: list[dict[str, Any]] = []
        self._embedder = None

    def _ensure_index(self) -> bool:
        """Load or build the FAISS index if needed."""
        index_file = self._index_path / "index.faiss"
        meta_file = self._index_path / "metadata.json"

        if not index_file.exists():
            logger.info("faiss_index_not_found_building_now")
            indexer = RAGIndexer()
            indexer.build_index()

        if not index_file.exists():
            logger.error("faiss_index_build_failed")
            return False

        if self._index is None:
            import faiss
            self._index = faiss.read_index(str(index_file))
            self._metadata = json.loads(meta_file.read_text())
            logger.info("faiss_index_loaded", chunks=len(self._metadata))

        if self._embedder is None:
            # Shared with the FAQ semantic cache — one model load per process.
            from src.core.embeddings import get_embedder
            self._embedder = get_embedder()

        return True

    def warm_up(self) -> bool:
        """Eagerly load the FAISS index and embedding model, and run one encode.

        Call this once at server startup so the heavy sentence-transformers /
        torch import and model load are paid before the first user query rather
        than on it. The throwaway encode matters as much as the load: torch's
        first inference initialises lazily and costs ~300ms, which would
        otherwise land on the first query — and on the FAQ cache's first
        semantic lookup, the one path that is supposed to be instant.
        """
        if not self._ensure_index():
            return False
        try:
            self._embedder.encode(["warm up"])
        except Exception as exc:  # noqa: BLE001 — warm-up is best effort
            logger.warning("embedder_warmup_encode_failed", error=str(exc))
        return True

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return top-k most relevant chunks for the query."""
        import faiss
        import numpy as np

        if not self._ensure_index():
            return []

        k = top_k or self._top_k
        query_embedding = self._embedder.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search more than k so we can filter
        search_k = min(k * 3, len(self._metadata))
        distances, indices = self._index.search(query_embedding, search_k)

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._metadata[idx]
            if category_filter and chunk.get("category") != category_filter:
                continue
            results.append({
                **chunk,
                "score": float(dist),
            })
            if len(results) >= k:
                break

        return results

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> str:
        """Return formatted context string for LLM prompt injection."""
        return format_context(self.search(query, top_k=top_k, category_filter=category_filter))


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Retrieved chunks as the context block agents put in the prompt (both backends)."""
    return "\n\n---\n\n".join(
        f"[Source {i}: {chunk['title']} ({chunk['source']})]\n{chunk['text']}"
        for i, chunk in enumerate(chunks, 1)
    )


def rag_backend_name() -> str:
    """"pgvector" or "faiss", resolving the "auto" setting."""
    settings = get_settings()
    backend = settings.rag.backend
    if backend == "auto":
        return "pgvector" if settings.database_url else "faiss"
    if backend == "pgvector" and not settings.database_url:
        raise RuntimeError("rag.backend is 'pgvector' but DATABASE_URL is not set")
    return backend


@lru_cache(maxsize=1)
def get_retriever() -> Any:
    """The process-wide retriever: pgvector hybrid search, or the FAISS file index."""
    if rag_backend_name() == "pgvector":
        from src.rag.pgvector import PgVectorRetriever

        return PgVectorRetriever()
    return RAGRetriever()
