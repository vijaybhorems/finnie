"""Shared sentence-transformers embedder.

The RAG retriever and the FAQ semantic cache both need query embeddings from the
same model. Loading ``SentenceTransformer`` twice costs ~90 MB of RSS and a
second model load at startup, so both go through this one lru_cached factory.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    """Return the process-wide SentenceTransformer instance."""
    from sentence_transformers import SentenceTransformer

    model_name = get_settings().embeddings.model
    logger.info("embedder_loading", model=model_name)
    return SentenceTransformer(model_name)


def embed(texts: list[str], normalize: bool = True) -> np.ndarray:
    """Embed `texts` as float32; L2-normalised so a dot product is cosine."""
    vectors = get_embedder().encode(texts).astype(np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
    return vectors


def embed_one(text: str, normalize: bool = True) -> np.ndarray:
    """Embed a single string, returning a 1-D vector."""
    return embed([text], normalize=normalize)[0]
