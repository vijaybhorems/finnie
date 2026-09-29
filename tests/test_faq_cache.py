"""Tests for the semantic FAQ cache and its graph nodes.

The cache must be generous with paraphrases and strict about anything that
could date an answer — those two properties are what these tests pin down.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from langchain_core.messages import HumanMessage

from src.core.state import AgentType, FinancialData, FinnieState, UserProfile
from src.utils.semantic_cache import (
    SemanticFAQCache,
    current_tax_year,
    normalize_query,
    query_hash,
)
from src.workflow.faq_cache import (
    faq_cache_node,
    faq_cache_write_node,
    route_after_faq_cache,
)

# Deterministic stand-in for MiniLM: near-identical vectors for paraphrases,
# orthogonal ones for unrelated questions. Keeps the model off the test path.
_VECTORS = {
    "what is a p e ratio": [1.0, 0.0, 0.0],
    "whats the p e ratio": [0.995, 0.1, 0.0],       # cosine ~0.995 -> hit
    "explain price to earnings": [0.8, 0.6, 0.0],   # cosine 0.80 -> miss
    "how do i harvest tax losses": [0.0, 1.0, 0.0],
}


def _fake_embed(text: str) -> np.ndarray:
    return np.asarray(_VECTORS.get(normalize_query(text), [0.0, 0.0, 1.0]), dtype=np.float32)


class _FakeCache:
    """In-memory stand-in for the Redis-backed Cache (same public surface)."""

    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, key): return self.store.get(key)
    def set(self, key, value, ttl=None): self.store[key] = value
    def delete(self, key): self.store.pop(key, None)
    def cache_key(self, *parts): return ":".join(["finnie", *parts])


@pytest.fixture
def cache(monkeypatch):
    fake = _FakeCache()
    monkeypatch.setattr("src.utils.semantic_cache.get_cache", lambda: fake)
    return SemanticFAQCache(embed_fn=_fake_embed)


def _state(query: str, **kwargs) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)] if query else [],
        user_profile=UserProfile(),
        financial_data=kwargs.pop("financial_data", FinancialData()),
        **kwargs,
    )


class TestNormalization:
    def test_punctuation_and_case_collapse(self):
        assert normalize_query("What is a P/E ratio?") == normalize_query("what is a p e ratio")

    def test_hash_is_stable_across_formatting(self):
        assert query_hash("What is a P/E ratio?") == query_hash("  what IS a p/e   ratio ")

    def test_apostrophes_collapse_rather_than_split_words(self):
        assert normalize_query("What's the P/E ratio?") == "whats the p e ratio"


class TestExactTier:
    def test_stores_and_serves_exact_match(self, cache):
        assert cache.set("What is a P/E ratio?", "It compares price to earnings.", "finance_qa")
        hit = cache.get("what is a p/e ratio")

        assert hit is not None
        assert hit["match_type"] == "exact"
        assert hit["answer"] == "It compares price to earnings."
        assert hit["agent"] == "finance_qa"

    def test_miss_on_unknown_query(self, cache):
        assert cache.get("something never asked") is None


class TestSemanticTier:
    def test_paraphrase_above_threshold_hits(self, cache):
        cache.set("What is a P/E ratio?", "It compares price to earnings.", "finance_qa")
        hit = cache.get("What's the P/E ratio?")

        assert hit is not None
        assert hit["match_type"] == "semantic"
        assert hit["similarity"] >= 0.92

    def test_related_but_below_threshold_misses(self, cache):
        cache.set("What is a P/E ratio?", "It compares price to earnings.", "finance_qa")
        assert cache.get("Explain price to earnings") is None

    def test_unrelated_query_misses(self, cache):
        cache.set("What is a P/E ratio?", "It compares price to earnings.", "finance_qa")
        assert cache.get("How do I harvest tax losses?") is None


class TestInvalidation:
    def test_knowledge_base_change_invalidates(self, cache):
        cache.set("What is a P/E ratio?", "old answer", "finance_qa")
        with patch("src.utils.semantic_cache.get_kb_version", return_value="different"):
            assert cache.get("What is a P/E ratio?") is None

    def test_tax_year_rollover_invalidates(self, cache):
        cache.set("What is a P/E ratio?", "2026 answer", "finance_qa")
        with patch("src.utils.semantic_cache.current_tax_year", return_value=current_tax_year() + 1):
            assert cache.get("What is a P/E ratio?") is None

    def test_clear_empties_the_cache(self, cache):
        cache.set("What is a P/E ratio?", "answer", "finance_qa")
        cache.clear()
        assert cache.get("What is a P/E ratio?") is None


class TestFailOpen:
    def test_embedding_error_is_a_miss_not_a_crash(self, cache):
        cache.set("What is a P/E ratio?", "answer", "finance_qa")

        def boom(_text):
            raise RuntimeError("model unavailable")

        cache._embed_fn = boom
        assert cache.get("What's the P/E ratio?") is None  # exact tier misses, semantic raises


class TestCacheNodes:
    """Graph-level behaviour: what gets served, and what is allowed to be stored."""

    @pytest.fixture(autouse=True)
    def _wire_cache(self, monkeypatch, cache):
        monkeypatch.setattr("src.workflow.faq_cache.get_faq_cache", lambda: cache)
        self.cache = cache

    def test_hit_short_circuits_the_turn(self):
        self.cache.set("What is a P/E ratio?", "Price over earnings.", "finance_qa")
        result = faq_cache_node(_state("What is a P/E ratio?"))

        assert result["cache_hit"] is True
        assert result["final_response"] == "Price over earnings."
        assert result["next_agent"] == AgentType.FINANCE_QA
        assert result["messages"]

    def test_miss_falls_through(self):
        result = faq_cache_node(_state("Something new"))
        assert result["cache_hit"] is False
        assert "final_response" not in result

    def test_blocklisted_query_is_never_served_from_cache(self):
        """A cached answer must not become a way around the guardrail."""
        self.cache.set("porn", "should never be served", "finance_qa")
        result = faq_cache_node(_state("show me some porn"))
        assert result["cache_hit"] is False

    def test_route_after_cache(self):
        hit = _state("q")
        hit.cache_hit = True
        assert route_after_faq_cache(hit) == "hit"
        assert route_after_faq_cache(_state("q")) == "miss"

    # ── write eligibility ────────────────────────────────────────────────────

    def test_writes_knowledge_grounded_answer(self):
        state = _state("What is a P/E ratio?")
        state.next_agent = AgentType.FINANCE_QA
        state.needs_macro = False
        state.final_response = "Price over earnings."

        faq_cache_write_node(state)
        assert self.cache.get("What is a P/E ratio?") is not None

    def test_never_writes_when_macro_data_was_used(self):
        state = _state("How do current rates affect bonds?")
        state.next_agent = AgentType.FINANCE_QA
        state.needs_macro = True
        state.final_response = "With the fed funds rate at 4.3%..."

        faq_cache_write_node(state)
        assert self.cache.get("How do current rates affect bonds?") is None

    def test_never_writes_on_the_legacy_path(self):
        """needs_macro=None means the agent fetched macro data unconditionally."""
        state = _state("What is a P/E ratio?")
        state.next_agent = AgentType.FINANCE_QA
        state.needs_macro = None
        state.final_response = "Price over earnings."

        faq_cache_write_node(state)
        assert self.cache.get("What is a P/E ratio?") is None

    def test_never_writes_when_live_market_data_was_used(self):
        state = _state("What is a P/E ratio?", financial_data=FinancialData(tickers=["AAPL"]))
        state.next_agent = AgentType.FINANCE_QA
        state.needs_macro = False
        state.final_response = "AAPL trades at ..."

        faq_cache_write_node(state)
        assert self.cache.get("What is a P/E ratio?") is None

    @pytest.mark.parametrize(
        "agent",
        [AgentType.MARKET_ANALYSIS, AgentType.NEWS_SYNTHESIZER, AgentType.PORTFOLIO],
    )
    def test_never_writes_for_live_data_agents(self, agent):
        state = _state("What is the market doing?")
        state.next_agent = agent
        state.needs_macro = False
        state.final_response = "Markets are up."

        faq_cache_write_node(state)
        assert self.cache.get("What is the market doing?") is None

    def test_does_not_rewrite_a_cache_hit(self):
        state = _state("What is a P/E ratio?")
        state.cache_hit = True
        state.next_agent = AgentType.FINANCE_QA
        state.needs_macro = False
        state.final_response = "Price over earnings."

        faq_cache_write_node(state)
        assert self.cache.get("What is a P/E ratio?") is None
