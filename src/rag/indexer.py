"""RAG indexer — chunks documents, creates embeddings, persists FAISS index."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence.split())
        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_words: list[str] = []
            overlap_count = 0
            for sent in reversed(current):
                words = sent.split()
                if overlap_count + len(words) <= overlap:
                    overlap_words.insert(0, sent)
                    overlap_count += len(words)
                else:
                    break
            current = overlap_words
            current_len = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_len += s_len

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) > 10]


class RAGIndexer:
    """Builds and persists a FAISS index from the knowledge base."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._index_path = self._settings.rag_index_path
        self._kb_path = self._settings.knowledge_base_path
        self._chunk_size = self._settings.rag.chunk_size
        self._overlap = self._settings.rag.chunk_overlap

    def _get_embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self._settings.embeddings.model)

    def _load_documents(self) -> list[dict[str, Any]]:
        """Load all .txt and .md files from the knowledge base."""
        documents: list[dict[str, Any]] = []
        kb_path = Path(self._kb_path)
        if not kb_path.exists():
            logger.warning("knowledge_base_not_found", path=str(kb_path))
            return documents

        for file_path in kb_path.rglob("*.txt"):
            try:
                text = file_path.read_text(encoding="utf-8")
                documents.append({
                    "source": str(file_path.relative_to(kb_path)),
                    "category": file_path.parent.name,
                    "title": file_path.stem.replace("_", " ").title(),
                    "text": text,
                })
            except Exception as exc:
                logger.warning("document_load_error", file=str(file_path), error=str(exc))

        for file_path in kb_path.rglob("*.md"):
            try:
                text = file_path.read_text(encoding="utf-8")
                # Strip markdown headers for cleaner chunking
                clean = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
                documents.append({
                    "source": str(file_path.relative_to(kb_path)),
                    "category": file_path.parent.name,
                    "title": file_path.stem.replace("_", " ").title(),
                    "text": clean,
                })
            except Exception as exc:
                logger.warning("document_load_error", file=str(file_path), error=str(exc))

        logger.info("documents_loaded", count=len(documents))
        return documents

    def build_index(self, force: bool = False) -> None:
        """Build and save the FAISS index. Skips if already built unless force=True."""
        import faiss
        import numpy as np

        index_file = self._index_path / "index.faiss"
        meta_file = self._index_path / "metadata.json"

        if index_file.exists() and meta_file.exists() and not force:
            logger.info("faiss_index_already_exists_skipping_build")
            return

        documents = self._load_documents()
        if not documents:
            logger.warning("no_documents_to_index")
            return

        # Chunk documents
        all_chunks: list[dict[str, Any]] = []
        for doc in documents:
            chunks = _chunk_text(doc["text"], self._chunk_size, self._overlap)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id": f"{doc['source']}_{i}",
                    "source": doc["source"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "text": chunk,
                })

        logger.info("chunks_created", count=len(all_chunks))

        # Embed
        embedder = self._get_embedder()
        texts = [c["text"] for c in all_chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = embeddings.astype(np.float32)

        # Normalise for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalisation
        index.add(embeddings)

        # Persist
        self._index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_file))
        meta_file.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2))

        logger.info("faiss_index_built", chunks=len(all_chunks), dimension=dimension)
