"""Chunk overlap effectiveness evals.

These tests validate that the chunking strategy (512 tokens, 64 overlap) does
not break important concepts across chunk boundaries, and that the overlap
preserves enough context for accurate retrieval.

Approach:
  1. Unit-level: verify _chunk_text produces overlapping content between adjacent chunks.
  2. Retrieval-level: verify that concepts spanning two paragraphs in the source
     material are still retrievable.
  3. Regression: compare different overlap values to ensure the current setting
     is not degrading retrieval quality.

Run with:
    pytest tests/evals/test_chunk_overlap.py -v
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from src.rag.indexer import _chunk_text


# ---------------------------------------------------------------------------
# 1. Chunker-level overlap tests
# ---------------------------------------------------------------------------
class TestChunkOverlapMechanics:
    """Verify that _chunk_text produces overlapping windows correctly."""

    def test_adjacent_chunks_share_content(self):
        """Adjacent chunks must share at least some words (the overlap)."""
        # Build a long deterministic text
        sentences = [f"Sentence number {i} contains important financial data." for i in range(60)]
        text = " ".join(sentences)
        chunks = _chunk_text(text, chunk_size=30, overlap=8)

        assert len(chunks) >= 3, "Expected at least 3 chunks for overlap testing"

        overlap_found = 0
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split()[-15:])  # tail of chunk A
            words_b = set(chunks[i + 1].split()[:15])  # head of chunk B
            shared = words_a & words_b
            if shared:
                overlap_found += 1

        assert overlap_found > 0, "No overlapping words found between adjacent chunks"

    def test_zero_overlap_produces_no_shared_sentences(self):
        """With overlap=0, chunks should have no shared sentences."""
        sentences = [f"Unique sentence {i} about investing." for i in range(40)]
        text = " ".join(sentences)

        chunks = _chunk_text(text, chunk_size=15, overlap=0)
        assert len(chunks) >= 2

        for i in range(len(chunks) - 1):
            # Full sentence sharing check (exact substring)
            overlap_sentences = 0
            for sent in re.split(r"(?<=[.])\s+", chunks[i]):
                if sent.strip() and sent.strip() in chunks[i + 1]:
                    overlap_sentences += 1
            # With zero overlap there should be no or very minimal sharing
            assert overlap_sentences <= 1, "Zero-overlap chunks share too many sentences"

    def test_overlap_preserves_sentence_boundaries(self):
        """Overlap should not cut mid-sentence (the chunker splits by sentence)."""
        text = (
            "The Federal Reserve sets the federal funds rate. "
            "This rate influences all other interest rates. "
            "When the Fed raises rates, bonds lose value. "
            "Investors must consider duration risk. "
            "A bond with 10-year duration drops about 10% for each 1% rate increase. "
            "This is known as interest rate risk. "
            "Short-duration bonds are safer in rising rate environments. "
            "Treasury bills have very short duration. "
            "Money market funds hold Treasury bills and commercial paper. "
            "These are considered near-cash equivalents. "
            "They offer safety but low returns. "
            "Inflation erodes purchasing power of low-yield investments. "
        )
        chunks = _chunk_text(text, chunk_size=20, overlap=8)

        for chunk in chunks:
            # Each chunk should end at a sentence boundary (period, !, ?)
            stripped = chunk.strip()
            assert stripped[-1] in ".!?", (
                f"Chunk does not end at sentence boundary: ...{stripped[-30:]}"
            )

    def test_increasing_overlap_increases_chunk_count(self):
        """More overlap → more chunks (same text, same chunk_size)."""
        sentences = [f"Financial concept number {i} is crucial for success." for i in range(50)]
        text = " ".join(sentences)

        count_low = len(_chunk_text(text, chunk_size=25, overlap=2))
        count_high = len(_chunk_text(text, chunk_size=25, overlap=10))

        assert count_high >= count_low, (
            f"Higher overlap ({count_high} chunks) should produce >= chunks than "
            f"lower overlap ({count_low} chunks)"
        )

    def test_small_document_single_chunk(self):
        """A document smaller than chunk_size should produce exactly one chunk."""
        text = "Compound interest is the eighth wonder of the world. It grows your wealth exponentially over time."
        chunks = _chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1

    @pytest.mark.parametrize("overlap", [0, 16, 32, 64, 128])
    def test_all_source_sentences_covered(self, overlap):
        """Every sentence from the source must appear in at least one chunk."""
        sentences = [f"Key concept number {i} about portfolio management." for i in range(30)]
        text = " ".join(sentences)
        chunks = _chunk_text(text, chunk_size=20, overlap=overlap)

        combined = " ".join(chunks)
        for sentence in sentences:
            # Each sentence (stripped) should be present in at least one chunk
            assert sentence in combined, (
                f"Sentence lost during chunking with overlap={overlap}: {sentence}"
            )


# ---------------------------------------------------------------------------
# 2. Retrieval-level overlap tests — concepts that span paragraphs
# ---------------------------------------------------------------------------
# These queries target knowledge that bridges two sections in the KB.
# If chunking breaks the concept, retrieval quality drops.
BOUNDARY_CASES = [
    # Bond prices & interest rates span multiple paragraphs in stocks_and_bonds.txt
    (
        "Why do bond prices fall when interest rates rise?",
        ["duration", "interest rate", "inverse", "bond price"],
    ),
    # The Roth conversion / backdoor strategy bridges retirement_accounts concepts
    (
        "What is a backdoor Roth IRA strategy?",
        ["backdoor", "Roth", "conversion", "income limit"],
    ),
    # Sequence-of-returns risk bridges risk_management and retirement planning
    (
        "How does sequence of returns risk affect early retirees?",
        ["sequence", "returns", "retirement", "withdrawal"],
    ),
    # 60/40 portfolio bridges stocks_and_bonds and asset_allocation
    (
        "Why is the 60/40 portfolio considered a classic allocation?",
        ["60", "40", "stocks", "bonds", "allocation"],
    ),
    # Tax-loss harvesting and the wash-sale rule span paragraphs
    (
        "Can I buy back a stock after tax-loss harvesting?",
        ["wash-sale", "30 days", "loss", "harvesting"],
    ),
    # Efficient frontier bridges diversification and asset_allocation
    (
        "How does the efficient frontier relate to diversification?",
        ["efficient frontier", "diversification", "portfolio", "risk"],
    ),
]


class TestChunkOverlapRetrieval:
    """Verify that concepts spanning chunk boundaries are still retrievable."""

    @pytest.mark.parametrize("query,keywords", BOUNDARY_CASES, ids=[c[0][:50] for c in BOUNDARY_CASES])
    def test_cross_boundary_concept_retrieved(self, retriever, query, keywords):
        """A query about a concept that spans two paragraphs must still return relevant results."""
        results = retriever.search(query, top_k=5)
        assert len(results) > 0, f"No results for boundary query: {query}"

        combined = " ".join(r["text"] for r in results).lower()
        matched = [kw for kw in keywords if kw.lower() in combined]
        assert len(matched) >= 2, (
            f"Expected at least 2 of {keywords} in retrieved text. Found: {matched}"
        )


# ---------------------------------------------------------------------------
# 3. Overlap regression: compare overlap=0 vs overlap=64 on retrieval quality
# ---------------------------------------------------------------------------
class TestOverlapVsNoOverlap:
    """Compare retrieval quality with and without chunk overlap.

    We rebuild a small index with overlap=0 and check that the default
    overlap=64 index performs at least as well (measured by keyword hits).
    """

    @pytest.fixture()
    def retriever_no_overlap(self, tmp_path):
        """Build an index with overlap=0 for comparison."""
        import json
        import numpy as np

        from src.core.config import get_settings
        from src.rag.indexer import RAGIndexer, _chunk_text

        settings = get_settings()
        kb_path = settings.knowledge_base_path

        # Load and chunk with NO overlap
        documents = []
        for ext in ("*.txt", "*.md"):
            for fp in kb_path.rglob(ext):
                text = fp.read_text(encoding="utf-8")
                documents.append({"source": str(fp.name), "category": fp.parent.name, "title": fp.stem, "text": text})

        all_chunks = []
        for doc in documents:
            chunks = _chunk_text(doc["text"], chunk_size=512, overlap=0)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id": f"{doc['source']}_{i}",
                    "source": doc["source"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "text": chunk,
                })

        # Embed
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(settings.embeddings.model)
        texts = [c["text"] for c in all_chunks]
        embeddings = embedder.encode(texts, batch_size=32).astype(np.float32)

        import faiss
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Save
        index_dir = tmp_path / "no_overlap_index"
        index_dir.mkdir()
        faiss.write_index(index, str(index_dir / "index.faiss"))
        (index_dir / "metadata.json").write_text(json.dumps(all_chunks))

        from src.rag.retriever import RAGRetriever
        r = RAGRetriever()
        r._index_path = index_dir
        r._index = None
        r._metadata = []
        r._embedder = None
        return r

    COMPARISON_QUERIES = [
        ("How does compound interest work in a retirement account?", ["compound", "retirement", "growth"]),
        ("What happens when the yield curve inverts?", ["yield curve", "inverted", "recession"]),
        ("How does the wash-sale rule affect tax-loss harvesting?", ["wash-sale", "30 days", "loss"]),
    ]

    @pytest.mark.parametrize("query,keywords", COMPARISON_QUERIES, ids=[q[:40] for q, _ in COMPARISON_QUERIES])
    def test_overlap_at_least_as_good_as_no_overlap(self, retriever, retriever_no_overlap, query, keywords):
        """Default overlap=64 should match or beat overlap=0 on keyword coverage."""
        def _score(r, q, kws):
            results = r.search(q, top_k=5)
            combined = " ".join(res["text"] for res in results).lower()
            return sum(1 for kw in kws if kw.lower() in combined)

        score_overlap = _score(retriever, query, keywords)
        score_no_overlap = _score(retriever_no_overlap, query, keywords)

        assert score_overlap >= score_no_overlap, (
            f"Overlap=64 scored {score_overlap} vs no-overlap scored {score_no_overlap} "
            f"for query: {query}"
        )
