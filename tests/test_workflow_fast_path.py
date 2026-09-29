"""End-to-end tests for the phase-1 fast path through the compiled graph.

Proves the two properties the phase is judged on: a miss costs one classifier
call plus one agent call (down from two classifier calls), and an immediate
repeat costs none at all.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.state import AgentType, FinnieState
from src.utils.semantic_cache import SemanticFAQCache, normalize_query
from src.workflow.classify import Verdict

_VECTORS = {
    "what is a p e ratio": [1.0, 0.0, 0.0],
    "whats a p e ratio": [0.99, 0.141, 0.0],  # paraphrase, cosine ~0.99
    "what is the market doing today": [0.0, 1.0, 0.0],
}


def _fake_embed(text: str) -> np.ndarray:
    return np.asarray(_VECTORS.get(normalize_query(text), [0.0, 0.0, 1.0]), dtype=np.float32)


class _FakeCache:
    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, key): return self.store.get(key)
    def set(self, key, value, ttl=None): self.store[key] = value
    def delete(self, key): self.store.pop(key, None)
    def cache_key(self, *parts): return ":".join(["finnie", *parts])


class _StubAgent:
    """Stands in for a real agent node; records how often it ran."""

    def __init__(self, response: str = "A P/E ratio compares price to earnings.") -> None:
        self.calls = 0
        self._response = response

    def run(self, state: FinnieState) -> dict:
        from langchain_core.messages import AIMessage

        self.calls += 1
        return {
            "messages": [AIMessage(content=self._response, name="Finance Q&A Agent")],
            "final_response": self._response,
        }


@pytest.fixture
def fast_path(monkeypatch):
    """Graph with a stub agent, a fake cache and a scripted classifier."""
    from src.workflow.graph import build_graph

    cache = SemanticFAQCache(embed_fn=_fake_embed)
    monkeypatch.setattr("src.utils.semantic_cache.get_cache", lambda: _FakeCache())
    cache._cache = _FakeCache()
    monkeypatch.setattr("src.workflow.faq_cache.get_faq_cache", lambda: cache)

    agent = _StubAgent()
    monkeypatch.setattr("src.workflow.graph._get_agent", lambda _type: agent)

    structured = MagicMock()
    structured.invoke.return_value = Verdict(
        on_topic=True, agent="finance_qa", needs_macro=False, reason="conceptual"
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr("src.workflow.classify.get_llm", lambda **kw: llm)

    build_graph.cache_clear()
    return {"graph": build_graph(), "agent": agent, "classifier": structured, "cache": cache}


def _run(graph, query: str) -> dict:
    from langchain_core.messages import HumanMessage

    return graph.invoke(FinnieState(messages=[HumanMessage(content=query)]))


class TestFastPath:
    def test_miss_runs_classifier_then_agent(self, fast_path):
        result = _run(fast_path["graph"], "What is a P/E ratio?")

        assert result["cache_hit"] is False
        assert result["final_response"].startswith("A P/E ratio")
        assert fast_path["classifier"].invoke.call_count == 1
        assert fast_path["agent"].calls == 1

    def test_repeat_is_served_with_zero_llm_calls(self, fast_path):
        graph = fast_path["graph"]
        _run(graph, "What is a P/E ratio?")

        fast_path["classifier"].invoke.reset_mock()
        result = _run(graph, "What is a P/E ratio?")

        assert result["cache_hit"] is True
        assert result["final_response"].startswith("A P/E ratio")
        assert fast_path["classifier"].invoke.call_count == 0  # no classifier call
        assert fast_path["agent"].calls == 1                   # agent did not run again

    def test_paraphrase_is_served_from_cache(self, fast_path):
        graph = fast_path["graph"]
        _run(graph, "What is a P/E ratio?")

        result = _run(graph, "What's a P/E ratio?")
        assert result["cache_hit"] is True
        assert fast_path["agent"].calls == 1

    def test_live_data_answer_is_not_cached(self, fast_path, monkeypatch):
        """A market answer must be recomputed every time."""
        structured = MagicMock()
        structured.invoke.return_value = Verdict(
            on_topic=True, agent="market_analysis", needs_macro=False, reason="live"
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        monkeypatch.setattr("src.workflow.classify.get_llm", lambda **kw: llm)

        graph = fast_path["graph"]
        _run(graph, "What is the market doing today?")
        result = _run(graph, "What is the market doing today?")

        assert result["cache_hit"] is False
        assert fast_path["agent"].calls == 2

    def test_off_topic_never_reaches_an_agent(self, fast_path, monkeypatch):
        structured = MagicMock()
        structured.invoke.return_value = Verdict(
            on_topic=False, agent="finance_qa", reason="cooking"
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        monkeypatch.setattr("src.workflow.classify.get_llm", lambda **kw: llm)

        result = _run(fast_path["graph"], "Give me a lasagna recipe")

        assert result["final_response"]
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE
        assert fast_path["agent"].calls == 0

    def test_blocklisted_query_costs_no_llm_call(self, fast_path):
        result = _run(fast_path["graph"], "show me some porn")

        assert result["next_agent"] == AgentType.OUT_OF_SCOPE
        assert fast_path["classifier"].invoke.call_count == 0
        assert fast_path["agent"].calls == 0


class TestLegacyPathStillWorks:
    """Both flags off must reproduce the original guardrail → router → agent graph."""

    def test_legacy_graph_shape(self, monkeypatch):
        from src.core.config import get_settings
        from src.workflow.graph import build_graph

        settings = get_settings()
        monkeypatch.setattr(settings.fast_path, "merged_classifier", False)
        monkeypatch.setattr(settings.fast_path.faq_cache, "enabled", False)

        build_graph.cache_clear()
        nodes = set(build_graph().get_graph().nodes.keys())

        assert {"guardrail", "router"} <= nodes
        assert "classify" not in nodes
        assert "faq_cache" not in nodes


class _Chunk:
    """Minimal stand-in for an AIMessageChunk."""

    def __init__(self, text: str) -> None:
        self.content = text


class _ScriptedGraph:
    """Graph whose .stream() replays a fixed sequence of LangGraph events."""

    def __init__(self, events) -> None:
        self._events = events

    def stream(self, _state, stream_mode=None):
        yield from self._events


class TestStreamWorkflow:
    def test_yields_agent_tokens_and_flushes_the_disclaimer(self, monkeypatch):
        from src.workflow.graph import stream_workflow

        events = [
            # The classifier's own tokens must not reach the user.
            ("messages", (_Chunk("{on_topic"), {"langgraph_node": "classify"})),
            ("messages", (_Chunk("A P/E ratio "), {"langgraph_node": "finance_qa"})),
            ("messages", (_Chunk("compares price to earnings."), {"langgraph_node": "finance_qa"})),
            ("values", {
                "final_response": "A P/E ratio compares price to earnings.\n\n---\nDisclaimer",
                "next_agent": AgentType.FINANCE_QA,
                "cache_hit": False,
            }),
        ]
        monkeypatch.setattr("src.workflow.graph.build_graph", lambda: _ScriptedGraph(events))

        sink: dict = {}
        chunks = list(stream_workflow("What is a P/E ratio?", sink=sink))

        assert "".join(chunks) == "A P/E ratio compares price to earnings.\n\n---\nDisclaimer"
        assert "{on_topic" not in "".join(chunks)
        assert sink["final_response"].endswith("Disclaimer")
        assert sink["agent_used"] == "finance_qa"

    def test_cache_hit_emits_the_whole_answer(self, monkeypatch):
        from src.workflow.graph import stream_workflow

        events = [
            ("values", {
                "final_response": "Cached answer.",
                "next_agent": AgentType.FINANCE_QA,
                "cache_hit": True,
            }),
        ]
        monkeypatch.setattr("src.workflow.graph.build_graph", lambda: _ScriptedGraph(events))

        sink: dict = {}
        chunks = list(stream_workflow("What is a P/E ratio?", sink=sink))

        assert "".join(chunks) == "Cached answer."
        assert sink["cache_hit"] is True

    def test_stream_error_is_surfaced_not_raised(self, monkeypatch):
        from src.workflow.graph import stream_workflow

        class _Boom:
            def stream(self, *_a, **_kw):
                raise RuntimeError("model unavailable")
                yield  # pragma: no cover

        monkeypatch.setattr("src.workflow.graph.build_graph", lambda: _Boom())

        sink: dict = {}
        chunks = list(stream_workflow("anything", sink=sink))

        assert "error" in "".join(chunks).lower()
        assert sink["agent_used"] == "error"
