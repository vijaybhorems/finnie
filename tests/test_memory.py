"""Phase 5: long-term memory — extraction, recall, privacy, deletion.

Runs on the in-memory store always, and on Postgres too when
FINNIE_TEST_DATABASE_URL is set. The extractor's model call is scripted, and
the store embeds with tests/conftest.py's deterministic hash_embed; the real
model and real extraction are exercised by tests/evals/test_memory_evals.py.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.core.state import FinnieState
from src.memory.extractor import Extraction, ExtractedFact, is_sensitive, may_contain_facts
from src.persistence.identity import user_id_from_claims
from src.workflow.classify import Verdict

_PG_URL = os.environ.get("FINNIE_TEST_DATABASE_URL", "")


@pytest.fixture(params=[
    "memory",
    pytest.param("postgres", marks=pytest.mark.skipif(not _PG_URL, reason="set FINNIE_TEST_DATABASE_URL")),
])
def backend(request, monkeypatch):
    from src.core.config import get_settings
    from src.persistence.backend import reset_persistence
    from src.workflow.graph import build_graph

    monkeypatch.setenv("DATABASE_URL", _PG_URL if request.param == "postgres" else "")
    get_settings.cache_clear()
    reset_persistence()
    build_graph.cache_clear()
    monkeypatch.setattr(get_settings().fast_path.faq_cache, "enabled", False)
    yield request.param
    reset_persistence()
    build_graph.cache_clear()


def _new_user() -> str:
    return user_id_from_claims(sub=str(uuid.uuid4().int)[:21], email=None)


def _user(user_id: str):
    from src.persistence.user_data import UserData

    return UserData(user_id)


def _fact(fact: str, category: str = "goal", confidence: float = 0.95, expires_on=None) -> ExtractedFact:
    return ExtractedFact(fact=fact, category=category, confidence=confidence, expires_on=expires_on)


@pytest.fixture
def extractor(monkeypatch):
    """Script what the extraction model returns; count its calls."""
    structured = MagicMock()
    structured.invoke.return_value = Extraction(facts=[])
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr("src.memory.extractor.get_llm", lambda **_: llm)

    def script(*facts: ExtractedFact) -> MagicMock:
        structured.invoke.return_value = Extraction(facts=list(facts))
        return structured

    script.calls = structured.invoke
    return script


# ── Extractor guards ───────────────────────────────────────────────────────

class TestExtractorGuards:
    @pytest.mark.parametrize("message,expected", [
        ("What is a P/E ratio?", False),
        ("How do index funds work?", False),
        ("I'm 34 and saving for a house", True),
        ("our kids start college in 2031", True),
        ("Should I buy bonds?", True),
    ])
    def test_first_person_gate(self, message, expected):
        assert may_contain_facts(message) is expected

    def test_no_first_person_means_no_model_call(self, extractor):
        from src.memory.extractor import extract_facts

        assert extract_facts("What is dollar-cost averaging?") == []
        extractor.calls.assert_not_called()

    @pytest.mark.parametrize("fact,sensitive", [
        ("Account number 123456789 at the bank.", True),
        ("SSN is 123-45-6789.", True),
        ("Card 4111 1111 1111 1111.", True),
        ("Reach them at someone@example.com.", True),
        ("Their password is hunter2.", True),
        ("Is 34 years old.", False),
        ("Saving $40,000 for a house deposit by 2028.", False),
        ("Has a 401k and a Roth IRA.", False),
    ])
    def test_sensitive_filter(self, fact, sensitive):
        assert is_sensitive(fact) is sensitive

    def test_model_error_yields_no_facts(self, monkeypatch):
        from src.memory.extractor import extract_facts

        llm = MagicMock()
        llm.with_structured_output.side_effect = RuntimeError("API down")
        monkeypatch.setattr("src.memory.extractor.get_llm", lambda **_: llm)
        assert extract_facts("I am saving for a house") == []


# ── remember_turn ──────────────────────────────────────────────────────────

class TestRememberTurn:
    def test_saves_confident_facts(self, backend, extractor):
        from src.memory.service import remember_turn

        user_id = _new_user()
        extractor(_fact("Saving for a house deposit, target 2028."), _fact("Is 34 years old.", "situation"))
        saved = remember_turn(user_id, "I'm 34 and saving for a house deposit by 2028")

        assert set(saved) == {"Saving for a house deposit, target 2028.", "Is 34 years old."}
        assert {m["fact"] for m in _user(user_id).list_memories()} == set(saved)

    def test_drops_low_confidence_and_sensitive_facts(self, backend, extractor):
        from src.memory.service import remember_turn

        user_id = _new_user()
        extractor(
            _fact("Might retire early someday.", confidence=0.4),
            _fact("Account number 123456789 at the bank.", "situation"),
            _fact("Prefers index funds.", "preference"),
        )
        assert remember_turn(user_id, "I prefer index funds") == ["Prefers index funds."]

    def test_caps_facts_per_turn(self, backend, extractor, monkeypatch):
        from src.core.config import get_settings
        from src.memory.service import remember_turn

        monkeypatch.setattr(get_settings().memory, "max_facts_per_turn", 2)
        extractor(*[_fact(f"Distinct goal number {i} about {w}.") for i, w in enumerate(["bonds", "cars", "yachts"])])
        assert len(remember_turn(_new_user(), "I have three goals")) == 2

    def test_users_switch_off_means_no_extraction(self, backend, extractor):
        from src.memory.service import remember_turn

        user_id = _new_user()
        _user(user_id).set_memory_enabled(False)
        extractor(_fact("Saving for a house."))

        assert remember_turn(user_id, "I'm saving for a house") == []
        extractor.calls.assert_not_called()

    def test_global_flag_off(self, backend, extractor, monkeypatch):
        from src.core.config import get_settings
        from src.memory.service import remember_turn

        monkeypatch.setattr(get_settings().memory, "enabled", False)
        extractor(_fact("Saving for a house."))
        assert remember_turn(_new_user(), "I'm saving for a house") == []

    def test_runs_in_the_background(self, backend, extractor):
        from src.memory.service import schedule_remember_turn

        user_id = _new_user()
        extractor(_fact("Saving for a house deposit, target 2028."))
        future = schedule_remember_turn(user_id, "I'm saving for a house")
        assert future.result(timeout=10) == ["Saving for a house deposit, target 2028."]


# ── Storage behaviour ──────────────────────────────────────────────────────

class TestMemoryStore:
    def test_near_duplicate_replaces_older_fact(self, backend, monkeypatch):
        """With one axis per word, "…target 2029" vs "…target 2028" is 6/7 ≈ 0.86."""
        user = _user(_new_user())
        first = user.remember("Saving for a house deposit, target 2028.", "goal", 0.9)
        second = user.remember("Saving for a house deposit, target 2029.", "goal", 0.9)

        assert first == second
        assert [m["fact"] for m in user.list_memories()] == ["Saving for a house deposit, target 2029."]

    def test_same_words_in_another_category_are_kept_apart(self, backend):
        user = _user(_new_user())
        user.remember("Has a Roth IRA.", "situation", 0.9)
        user.remember("Has a Roth IRA.", "goal", 0.9)
        assert len(user.list_memories()) == 2

    def test_most_relevant_first_and_top_k(self, backend, monkeypatch):
        from src.core.config import get_settings

        monkeypatch.setattr(get_settings().memory, "top_k", 2)
        user = _user(_new_user())
        user.remember("Saving for a house deposit by 2028.", "goal", 0.9)
        user.remember("Prefers low cost index funds.", "preference", 0.9)
        user.remember("Has two children.", "situation", 0.9)

        relevant = user.relevant_memories("how much for a house deposit")
        assert len(relevant) == 2
        assert relevant[0]["fact"] == "Saving for a house deposit by 2028."

    def test_expired_memories_are_dropped(self, backend):
        user = _user(_new_user())
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        user.remember("Contract ends soon.", "situation", 0.9, expires_on=yesterday)
        user.remember("Is 34 years old.", "situation", 0.9)
        assert [m["fact"] for m in user.list_memories()] == ["Is 34 years old."]

    def test_forget_is_a_real_delete(self, backend):
        from src.persistence.backend import get_store

        user = _user(_new_user())
        memory_id = user.remember("Saving for a house.", "goal", 0.9)
        user.forget(memory_id)
        assert get_store().get(("users", user.user_id, "memories"), memory_id) is None
        assert user.list_memories() == []

    def test_forget_all(self, backend):
        user = _user(_new_user())
        for fact in ("Is 34.", "Has a 401k.", "Prefers ETFs."):
            user.remember(fact, "situation", 0.9)
        assert user.forget_all() == 3 and user.list_memories() == []

    def test_capped_per_user_oldest_evicted(self, backend, monkeypatch):
        from src.core.config import get_settings

        monkeypatch.setattr(get_settings().memory, "max_memories_per_user", 2)
        user = _user(_new_user())
        for fact in ("Owns a condo downtown.", "Drives an electric pickup.", "Collects vintage watches."):
            user.remember(fact, "situation", 0.9)
        assert [m["fact"] for m in user.list_memories()] == ["Collects vintage watches.", "Drives an electric pickup."]

    def test_switch_off_hides_memories_from_recall(self, backend):
        user = _user(_new_user())
        user.remember("Saving for a house.", "goal", 0.9)
        user.set_memory_enabled(False)
        assert user.relevant_memories("house") == []
        assert len(user.list_memories()) == 1      # kept until the user deletes it

    def test_profile_writes_do_not_embed(self, backend, monkeypatch):
        """Only memories are indexed; profile and holdings never pay for an embedding."""
        from tests.conftest import hash_embed

        calls = []
        monkeypatch.setattr("src.persistence.backend.embed_texts", lambda t: calls.append(t) or hash_embed(t))
        user = _user(_new_user())
        user.save_profile({"knowledge_level": "advanced"})
        user.save_holdings([{"ticker": "VTI", "shares": 1, "avg_cost": 1}])
        assert calls == []


# ── Recall in a turn: the multi-session eval, with a recording agent ───────

class RecordingAgent:
    def __init__(self) -> None:
        self.seen: list[FinnieState] = []

    def run(self, state: FinnieState) -> dict[str, Any]:
        self.seen.append(state.model_copy(deep=True))
        return {"messages": [AIMessage(content="answer", name="Finance Q&A Agent")], "final_response": "answer"}


@pytest.fixture
def agent(monkeypatch, backend) -> RecordingAgent:
    recorder = RecordingAgent()
    monkeypatch.setattr("src.workflow.graph._get_agent", lambda _type: recorder)
    structured = MagicMock()
    structured.invoke.return_value = Verdict(on_topic=True, agent="finance_qa", needs_macro=False, reason="t")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr("src.workflow.classify.get_llm", lambda **_: llm)
    return recorder


def _chat(user_id: str, message: str, new_session: bool = False) -> dict:
    from src.workflow.graph import run_workflow

    user = _user(user_id)
    thread = user.new_thread() if new_session else user.active_thread_id()
    return run_workflow(message, user_id=user_id, thread_id=thread)


class TestRecall:
    def test_goal_from_an_earlier_session_is_recalled(self, agent, extractor):
        """The design doc's exit criterion: stated once, recalled later without re-asking."""
        from src.memory.service import remember_turn

        user_id = _new_user()
        extractor(_fact("Saving for a house deposit, target 2028."))
        _chat(user_id, "I'm saving for a house deposit by 2028")
        remember_turn(user_id, "I'm saving for a house deposit by 2028")

        _chat(user_id, "How much should I keep in bonds for my house deposit?", new_session=True)

        later = agent.seen[-1]
        assert [m.content for m in later.messages] == ["How much should I keep in bonds for my house deposit?"]
        assert "Saving for a house deposit, target 2028." in [m["fact"] for m in later.user_profile.memories]

    def test_memories_reach_the_agent_prompt(self, backend):
        from unittest.mock import patch

        from src.core.state import UserProfile

        with patch("src.agents.finance_qa_agent.get_retriever"), \
             patch("src.agents.finance_qa_agent.FredClient"), \
             patch("src.agents.base_agent.get_llm"):
            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
        state = FinnieState(user_profile=UserProfile(memories=[{"fact": "Is 34 years old.", "noted_on": "2026-09-01"}]))

        context = agent._get_user_context_str(state)
        assert "Saved memories" in context and "- Is 34 years old. (noted 2026-09-01)" in context
        assert "Saved memories" not in agent._get_user_context_str(FinnieState())

    def test_deleted_memory_is_no_longer_recalled(self, agent):
        user_id = _new_user()
        memory_id = _user(user_id).remember("Saving for a house deposit, target 2028.", "goal", 0.9)
        _user(user_id).forget(memory_id)

        _chat(user_id, "How much for my house deposit?", new_session=True)
        assert agent.seen[-1].user_profile.memories == []

    def test_other_users_memories_never_appear(self, agent):
        alice, bob = _new_user(), _new_user()
        _user(alice).remember("Saving for a house deposit, target 2028.", "goal", 0.9)

        _chat(bob, "How much should I save for a house deposit?")
        assert agent.seen[-1].user_profile.memories == []

    def test_recall_failure_keeps_the_rest_of_the_profile(self, agent, monkeypatch):
        user_id = _new_user()
        _user(user_id).save_holdings([{"ticker": "VTI", "shares": 1, "avg_cost": 1}])
        monkeypatch.setattr(
            "src.persistence.user_data.UserData.relevant_memories",
            lambda self, q, k=None: (_ for _ in ()).throw(RuntimeError("index down")),
        )
        result = _chat(user_id, "What about my portfolio?")

        assert result["final_response"] == "answer"
        assert agent.seen[-1].user_profile.portfolio[0]["ticker"] == "VTI"


class TestFAQCacheStaysGeneric:
    """The FAQ cache is shared by every user; personalised answers must not enter it."""

    def _state(self, memories):
        from src.core.state import AgentType, UserProfile
        from langchain_core.messages import HumanMessage

        return FinnieState(
            messages=[HumanMessage(content="What is a P/E ratio?")],
            next_agent=AgentType.FINANCE_QA,
            needs_macro=False,
            final_response="P/E compares price to earnings.",
            user_profile=UserProfile(memories=memories),
        )

    def test_answer_with_memories_is_not_cached(self, monkeypatch):
        from src.workflow.faq_cache import faq_cache_write_node

        cache = MagicMock()
        monkeypatch.setattr("src.workflow.faq_cache.get_faq_cache", lambda: cache)
        faq_cache_write_node(self._state([{"fact": "Is 34 years old."}]))
        cache.set.assert_not_called()

    def test_generic_answer_is_still_cached(self, monkeypatch):
        from src.workflow.faq_cache import faq_cache_write_node

        cache = MagicMock()
        monkeypatch.setattr("src.workflow.faq_cache.get_faq_cache", lambda: cache)
        faq_cache_write_node(self._state([]))
        cache.set.assert_called_once()


# ── Postgres-only: exact search across many users ──────────────────────────

@pytest.mark.skipif(not _PG_URL, reason="set FINNIE_TEST_DATABASE_URL")
def test_a_users_memory_is_found_among_many_similar_ones(monkeypatch):
    """Every user's memories share one table, filtered by namespace. Pins the
    requirement that a user's own memory is found even when many other users
    hold closer matches to the query. (It passes with LangGraph's default HNSW
    too — its query filters by namespace before ranking — so this guards the
    behaviour, not the index choice; see _index_config.)"""
    from src.core.config import get_settings
    from src.persistence.backend import reset_persistence

    monkeypatch.setenv("DATABASE_URL", _PG_URL)
    get_settings.cache_clear()
    reset_persistence()
    try:
        for _ in range(120):
            _user(_new_user()).remember("Saving for a house deposit, target 2028.", "goal", 0.9)
        me = _user(_new_user())
        me.remember("Wants a house deposit fund by spring.", "goal", 0.9)

        found = me.relevant_memories("Saving for a house deposit, target 2028.")
        assert [m["fact"] for m in found] == ["Wants a house deposit fund by spring."]
    finally:
        reset_persistence()
