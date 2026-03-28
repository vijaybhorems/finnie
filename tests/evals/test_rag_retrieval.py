"""RAG retrieval quality evals.

These tests verify that the FAISS index returns *relevant* chunks
for representative financial queries across every knowledge-base category.
They use the real embedding model and the real FAISS index — no mocks.

Run with:
    pytest tests/evals/test_rag_retrieval.py -v
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Each case is (query, expected_category, expected_keywords).
# `expected_category` must appear in at least one of the top-k results.
# `expected_keywords` — at least one must appear in the retrieved text.
# ---------------------------------------------------------------------------
RETRIEVAL_CASES = [
    # Investing Basics
    (
        "What is compound interest and why does it matter?",
        "investing_basics",
        ["compound", "interest", "growth"],
    ),
    (
        "Explain the difference between stocks and bonds",
        "investing_basics",
        ["stock", "bond", "equity", "fixed income"],
    ),
    (
        "What is an ETF and how is it different from a mutual fund?",
        "investing_basics",
        ["etf", "index fund", "expense ratio"],
    ),
    (
        "What is dollar-cost averaging?",
        "investing_basics",
        ["dollar-cost averaging", "DCA", "dollar cost"],
    ),
    # Market Concepts
    (
        "What is RSI and how do traders use it?",
        "market_concepts",
        ["RSI", "overbought", "oversold", "relative strength"],
    ),
    (
        "How does the yield curve predict recessions?",
        "market_concepts",
        ["yield curve", "inverted", "recession"],
    ),
    (
        "What are leading economic indicators?",
        "market_concepts",
        ["leading", "indicator", "PMI", "consumer confidence"],
    ),
    (
        "What is the golden cross in technical analysis?",
        "market_concepts",
        ["golden cross", "moving average", "50-day", "200-day"],
    ),
    # Risk Management
    (
        "What is beta and how does it measure risk?",
        "risk_management",
        ["beta", "volatility", "systematic"],
    ),
    (
        "Explain the Sharpe ratio",
        "risk_management",
        ["Sharpe", "risk-adjusted", "volatility"],
    ),
    (
        "What is sequence of returns risk in retirement?",
        "risk_management",
        ["sequence", "returns", "retirement"],
    ),
    # Tax Accounts
    (
        "How does tax-loss harvesting work?",
        "tax_accounts",
        ["tax-loss harvesting", "wash-sale", "capital loss"],
    ),
    (
        "What are the contribution limits for a Roth IRA?",
        "tax_accounts",
        ["Roth", "IRA", "contribution", "limit"],
    ),
    (
        "What is an HSA and why is it called triple tax advantage?",
        "tax_accounts",
        ["HSA", "triple", "tax"],
    ),
    # Goal Planning
    (
        "What is the 4% rule in retirement planning?",
        "goal_planning",
        ["4%", "withdrawal", "retirement"],
    ),
    (
        "How do I set up an emergency fund?",
        "goal_planning",
        ["emergency fund", "3-6 months", "savings"],
    ),
    (
        "What is the FIRE movement?",
        "goal_planning",
        ["FIRE", "financial independence", "early retirement"],
    ),
    # Portfolio Management
    (
        "Why is diversification important in a portfolio?",
        "portfolio_management",
        ["diversification", "correlation", "risk"],
    ),
    (
        "What is asset allocation and the rule of 110?",
        "portfolio_management",
        ["asset allocation", "110", "stocks", "bonds"],
    ),
    (
        "How does rebalancing a portfolio work?",
        "portfolio_management",
        ["rebalancing", "drift", "allocation"],
    ),
]


class TestRAGRetrievalQuality:
    """Verify the retriever surfaces the right knowledge-base category and keywords."""

    @pytest.mark.parametrize("query,expected_category,keywords", RETRIEVAL_CASES, ids=[c[0][:50] for c in RETRIEVAL_CASES])
    def test_retrieval_hits_expected_category(self, retriever, query, expected_category, keywords):
        """At least one of the top-k results must come from the expected category."""
        results = retriever.search(query, top_k=5)
        assert len(results) > 0, f"No results returned for: {query}"

        categories = {r["category"] for r in results}
        assert expected_category in categories, (
            f"Expected category '{expected_category}' not found in top-5 results. "
            f"Got categories: {categories}"
        )

    @pytest.mark.parametrize("query,expected_category,keywords", RETRIEVAL_CASES, ids=[c[0][:50] for c in RETRIEVAL_CASES])
    def test_retrieval_contains_expected_keywords(self, retriever, query, expected_category, keywords):
        """The combined retrieved text must contain at least one expected keyword."""
        results = retriever.search(query, top_k=5)
        combined_text = " ".join(r["text"] for r in results).lower()

        matched = [kw for kw in keywords if kw.lower() in combined_text]
        assert matched, (
            f"None of {keywords} found in top-5 retrieved text for: {query}"
        )

    def test_category_filter_works(self, retriever):
        """Filtered search must only return chunks from the specified category."""
        results = retriever.search("tax-loss harvesting", top_k=5, category_filter="tax_accounts")
        for r in results:
            assert r["category"] == "tax_accounts", (
                f"Expected only tax_accounts results, got: {r['category']}"
            )

    def test_top_result_has_high_relevance_score(self, retriever):
        """The top result for a very specific query should score above a threshold."""
        results = retriever.search("What is compound interest?", top_k=1)
        assert len(results) > 0
        # Cosine similarity after L2 normalization; 0.35 is a reasonable floor
        assert results[0]["score"] > 0.35, (
            f"Top result score {results[0]['score']:.3f} is below 0.35 threshold"
        )

    def test_no_duplicate_chunks(self, retriever):
        """The same chunk should not appear twice in results."""
        results = retriever.search("What is diversification?", top_k=5)
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks found in results"


class TestCrossCategory:
    """Verify queries that span multiple topics return chunks from multiple categories."""

    def test_broad_query_returns_multiple_categories(self, retriever):
        """A broad query should pull from more than one category."""
        results = retriever.search(
            "How do taxes affect my retirement portfolio?", top_k=5
        )
        categories = {r["category"] for r in results}
        assert len(categories) >= 2, (
            f"Expected cross-category results, got only: {categories}"
        )
