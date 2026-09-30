"""Phase 3 persistence: identity, per-user data, persistent threads, isolation.

Every scenario runs against the in-memory backend, and against real Postgres
when FINNIE_TEST_DATABASE_URL is set — the Postgres run exercises the real
connection pool, the LangGraph migrations and jsonb, which the memory backend
cannot (jsonb rejects NaN; InMemoryStore does not).

    FINNIE_TEST_DATABASE_URL=postgresql://user@localhost:5432/finnie_test pytest tests/test_persistence.py

The cross-user tests are the leakage eval from the design doc: no path may
show one user another user's profile, holdings, plan or conversation.
"""
from __future__ import annotations

import math
import os
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.state import FinancialData, FinnieState, UserProfile, trim_history, turn_state_reset
from src.persistence.identity import user_id_from_claims
from src.workflow.classify import Verdict

_PG_URL = os.environ.get("FINNIE_TEST_DATABASE_URL", "")


@pytest.fixture(params=[
    "memory",
    pytest.param("postgres", marks=pytest.mark.skipif(not _PG_URL, reason="set FINNIE_TEST_DATABASE_URL")),
])
def backend(request, monkeypatch):
    """Point the real persistence code at the chosen backend."""
    from src.core.config import get_settings
    from src.persistence.backend import reset_persistence
    from src.workflow.graph import build_graph

    monkeypatch.setenv("DATABASE_URL", _PG_URL if request.param == "postgres" else "")
    get_settings.cache_clear()
    reset_persistence()
    build_graph.cache_clear()
    # No embedding model or FAQ cache in these tests: every turn reaches the agent.
    monkeypatch.setattr(get_settings().fast_path.faq_cache, "enabled", False)
    yield request.param
    reset_persistence()
    build_graph.cache_clear()


def _new_user() -> str:
    # Unique per test, so runs against a shared Postgres never see each other.
    return user_id_from_claims(sub=str(uuid.uuid4().int)[:21], email=None)


# ── Stubs: a recording agent and a scripted classifier ─────────────────────

class RecordingAgent:
    """Stands in for every agent; records exactly what state each turn saw."""

    def __init__(self) -> None:
        self.seen: list[FinnieState] = []
        self.set_live_data = False

    def run(self, state: FinnieState) -> dict[str, Any]:
        self.seen.append(state.model_copy(deep=True))
        turn = sum(1 for m in state.messages if m.type == "human")
        update: dict[str, Any] = {
            "messages": [AIMessage(content=f"answer {turn}", name="Finance Q&A Agent")],
            "final_response": f"answer {turn}",
        }
        if self.set_live_data:
            update["financial_data"] = FinancialData(tickers=["SPY"], price_data={"SPY": 500})
        return update


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


def _user_data(user_id: str):
    from src.persistence.user_data import UserData

    return UserData(user_id)


def _chat(user_id: str, thread_id: str, message: str) -> dict[str, Any]:
    from src.workflow.graph import run_workflow

    return run_workflow(message, user_id=user_id, thread_id=thread_id)


# ── Identity ───────────────────────────────────────────────────────────────

class TestIdentity:
    def test_prefers_google_sub(self):
        assert user_id_from_claims("1098765", "a.person@gmail.com") == "g-1098765"

    def test_falls_back_to_hashed_email_without_periods(self):
        uid = user_id_from_claims(None, "A.Person@Gmail.com")
        assert uid.startswith("e-") and "." not in uid and "person" not in uid
        assert uid == user_id_from_claims("", "a.person@gmail.com")  # case-insensitive

    def test_no_identity_is_an_error(self):
        with pytest.raises(ValueError):
            user_id_from_claims(None, None)


# ── UserData ───────────────────────────────────────────────────────────────

class TestUserData:
    def test_profile_defaults_then_round_trip(self, backend):
        data = _user_data(_new_user())
        assert data.get_profile() == {
            "knowledge_level": "beginner", "risk_tolerance": "moderate", "investment_horizon": "long",
        }
        data.save_profile({"knowledge_level": "advanced", "risk_tolerance": "yolo"})
        # Invalid values are ignored; valid ones persist across instances (a refresh).
        again = _user_data(data.user_id).get_profile()
        assert again["knowledge_level"] == "advanced"
        assert again["risk_tolerance"] == "moderate"

    def test_holdings_sanitised_nan_rows_dropped(self, backend):
        """st.data_editor yields NaN for blank cells; Postgres jsonb rejects NaN."""
        data = _user_data(_new_user())
        saved = data.save_holdings([
            {"ticker": " aapl ", "shares": 10, "avg_cost": 150.0},
            {"ticker": "MSFT", "shares": float("nan"), "avg_cost": 300.0},   # half-filled row
            {"ticker": None, "shares": 5, "avg_cost": 10.0},                 # blank ticker
            {"ticker": "VTI", "shares": 20, "avg_cost": float("nan")},       # cost not entered
        ])
        assert saved == [
            {"ticker": "AAPL", "shares": 10.0, "avg_cost": 150.0},
            {"ticker": "VTI", "shares": 20.0, "avg_cost": 0.0},
        ]
        assert _user_data(data.user_id).get_holdings() == saved

    def test_plan_with_non_finite_numbers_saves(self, backend):
        data = _user_data(_new_user())
        data.save_plan({"start_age": 30, "events": [{"kind": "inheritance", "amount": float("inf")}]})
        plan = _user_data(data.user_id).get_plan()
        assert plan["events"][0]["amount"] == 0.0

    def test_active_thread_is_stable_until_replaced(self, backend):
        user_id = _new_user()
        first = _user_data(user_id).active_thread_id()
        assert first.startswith(f"{user_id}:")
        assert _user_data(user_id).active_thread_id() == first     # a refresh keeps it
        replacement = _user_data(user_id).new_thread()
        assert replacement != first
        assert _user_data(user_id).active_thread_id() == replacement

    @pytest.mark.parametrize("bad", ["", "has.period"])
    def test_rejects_invalid_user_ids(self, backend, bad):
        with pytest.raises(ValueError):
            _user_data(bad)

    def test_to_user_profile_carries_holdings_and_plan(self, backend):
        data = _user_data(_new_user())
        data.save_holdings([{"ticker": "VTI", "shares": 3, "avg_cost": 200}])
        data.save_plan({"start_age": 40, "events": []})
        profile = data.to_user_profile()
        assert profile.user_id == data.user_id
        assert profile.portfolio == [{"ticker": "VTI", "shares": 3.0, "avg_cost": 200.0}]
        assert profile.goals == [{"type": "life_timeline", "start_age": 40, "events": []}]


# ── Persistent threads ─────────────────────────────────────────────────────

class TestPersistentThreads:
    def test_history_accumulates_across_turns(self, agent):
        user_id = _new_user()
        thread = _user_data(user_id).active_thread_id()
        _chat(user_id, thread, "first question")
        result = _chat(user_id, thread, "second question")

        assert result["final_response"] == "answer 2"
        second_turn_messages = [m.content for m in agent.seen[-1].messages]
        assert second_turn_messages == ["first question", "answer 1", "second question"]

    def test_conversation_survives_a_refresh(self, agent):
        from src.workflow.graph import load_conversation

        user_id = _new_user()
        thread = _user_data(user_id).active_thread_id()
        _chat(user_id, thread, "remember me")

        # A refresh: a new session resolves the same stored thread.
        same_thread = _user_data(user_id).active_thread_id()
        contents = [m.content for m in load_conversation(user_id, same_thread)]
        assert contents == ["remember me", "answer 1"]

    def test_restart_keeps_history_only_on_postgres(self, agent, backend):
        """The documented limit of the memory backend, pinned so it is never a surprise."""
        from src.persistence.backend import reset_persistence
        from src.workflow.graph import build_graph, load_conversation

        user_id = _new_user()
        thread = _user_data(user_id).active_thread_id()
        _chat(user_id, thread, "before restart")

        reset_persistence()          # a new process: fresh pool / fresh memory
        build_graph.cache_clear()

        survived = load_conversation(user_id, thread)
        if backend == "postgres":
            assert [m.content for m in survived] == ["before restart", "answer 1"]
        else:
            assert survived == []

    def test_per_turn_fields_do_not_leak_into_the_next_turn(self, agent):
        """A checkpointed thread carries every field forward unless it is reset."""
        user_id = _new_user()
        thread = _user_data(user_id).active_thread_id()

        agent.set_live_data = True                 # turn 1 looks like a market answer
        _chat(user_id, thread, "what is SPY doing")
        agent.set_live_data = False
        _chat(user_id, thread, "what is a P/E ratio")

        second = agent.seen[-1]
        assert second.financial_data == FinancialData()   # would block FAQ caching if stale
        assert second.final_response == ""
        assert second.cache_hit is False

    def test_reset_covers_every_non_message_field(self):
        assert set(turn_state_reset()) == set(FinnieState.model_fields) - {"messages"}

    def test_refusal_is_recorded_in_history(self, agent):
        from src.workflow.graph import load_conversation

        user_id = _new_user()
        thread = _user_data(user_id).active_thread_id()
        refused = _chat(user_id, thread, "show me some porn")   # blocklist, no LLM call
        _chat(user_id, thread, "ok, what is an ETF?")

        history = load_conversation(user_id, thread)
        assert [m.type for m in history] == ["human", "ai", "human", "ai"]
        assert history[1].content == refused["final_response"]

    def test_clear_deletes_the_saved_thread(self, agent):
        from src.workflow.graph import delete_conversation, load_conversation

        user_id = _new_user()
        thread = _user_data(user_id).active_thread_id()
        _chat(user_id, thread, "delete me")
        delete_conversation(user_id, thread)

        assert load_conversation(user_id, thread) == []


# ── Hydration ──────────────────────────────────────────────────────────────

class TestHydrate:
    def test_saved_data_reaches_the_agent(self, agent):
        """Guards the store-injection annotation: if LangGraph stopped injecting
        the store, hydrate would silently no-op and this would see defaults."""
        user_id = _new_user()
        data = _user_data(user_id)
        data.save_profile({"knowledge_level": "advanced", "risk_tolerance": "aggressive"})
        data.save_holdings([{"ticker": "NVDA", "shares": 2, "avg_cost": 400}])

        _chat(user_id, data.active_thread_id(), "how is my portfolio?")

        seen = agent.seen[-1].user_profile
        assert seen.knowledge_level == "advanced"
        assert seen.risk_tolerance == "aggressive"
        assert seen.portfolio == [{"ticker": "NVDA", "shares": 2.0, "avg_cost": 400.0}]

    def test_one_shot_callers_are_hydrated_too(self, agent):
        """Portfolio/Market/Goals tabs call run_workflow without a thread."""
        from src.workflow.graph import run_workflow

        user_id = _new_user()
        _user_data(user_id).save_holdings([{"ticker": "VTI", "shares": 1, "avg_cost": 1}])
        run_workflow("analyze my portfolio", user_id=user_id)

        assert agent.seen[-1].user_profile.portfolio[0]["ticker"] == "VTI"

    def test_store_failure_fails_open(self, agent, monkeypatch):
        user_id = _new_user()
        thread = _user_data(user_id).active_thread_id()
        monkeypatch.setattr(
            "src.workflow.hydrate.UserData.to_user_profile",
            lambda self: (_ for _ in ()).throw(RuntimeError("db down")),
        )
        result = _chat(user_id, thread, "still answer me")

        assert result["final_response"] == "answer 1"


# ── Cross-user isolation (the leakage eval) ────────────────────────────────

class TestCrossUserIsolation:
    def test_saved_data_is_invisible_to_other_users(self, backend):
        alice, bob = _new_user(), _new_user()
        a = _user_data(alice)
        a.save_profile({"knowledge_level": "advanced"})
        a.save_holdings([{"ticker": "AAPL", "shares": 1, "avg_cost": 1}])
        a.save_plan({"start_age": 50, "events": []})

        b = _user_data(bob)
        assert b.get_profile()["knowledge_level"] == "beginner"
        assert b.get_holdings() == []
        assert b.get_plan() is None
        assert b.active_thread_id() != a.active_thread_id()

    def test_conversations_and_profiles_do_not_cross(self, agent):
        alice, bob = _new_user(), _new_user()
        _user_data(alice).save_holdings([{"ticker": "AAPL", "shares": 1, "avg_cost": 1}])
        _chat(alice, _user_data(alice).active_thread_id(), "alice's secret question")
        _chat(bob, _user_data(bob).active_thread_id(), "bob's question")

        bob_turn = agent.seen[-1]
        assert [m.content for m in bob_turn.messages] == ["bob's question"]
        assert bob_turn.user_profile.portfolio == []

    @pytest.mark.parametrize("operation", ["run", "load", "delete"])
    def test_another_users_thread_is_refused(self, agent, operation):
        from src.workflow.graph import delete_conversation, load_conversation, run_workflow

        alice, bob = _new_user(), _new_user()
        alice_thread = _user_data(alice).active_thread_id()
        _chat(alice, alice_thread, "private")

        with pytest.raises(ValueError):
            if operation == "run":
                run_workflow("let me in", user_id=bob, thread_id=alice_thread)
            elif operation == "load":
                load_conversation(bob, alice_thread)
            else:
                delete_conversation(bob, alice_thread)

        # And alice's conversation is untouched.
        assert len(load_conversation(alice, alice_thread)) == 2

    def test_thread_without_user_is_refused(self, agent):
        from src.workflow.graph import run_workflow

        with pytest.raises(ValueError):
            run_workflow("hi", thread_id="g-123:abc")


# ── History window sent to the model ───────────────────────────────────────

class TestTrimHistory:
    def _history(self, turns: int) -> list:
        messages = []
        for i in range(turns):
            messages += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
        return messages

    def test_window_is_bounded_and_opens_on_a_user_turn(self):
        trimmed = trim_history(self._history(10) + [HumanMessage(content="now")], 5)
        assert trimmed[0].type == "human"
        assert len(trimmed) <= 5
        assert trimmed[-1].content == "now"

    def test_agent_sends_bounded_history(self, monkeypatch):
        from src.core.config import get_settings

        monkeypatch.setattr(get_settings().workflow, "max_history_messages", 4)
        with patch("src.agents.finance_qa_agent.get_retriever"), \
             patch("src.agents.finance_qa_agent.FredClient"), \
             patch("src.agents.base_agent.get_llm"):
            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
        state = FinnieState(messages=self._history(10) + [HumanMessage(content="now")])

        sent = agent._build_messages(state, "ctx")[1:]   # drop the system message
        assert len(sent) <= 4
        assert sent[0].type == "human" and sent[-1].content == "now"


# ── Page save rules (never save untouched example data) ────────────────────

class TestSaveIfChanged:
    @pytest.fixture(autouse=True)
    def _fake_streamlit(self, monkeypatch):
        monkeypatch.setattr("src.web_app.session.st.session_state", {}, raising=False)
        monkeypatch.setattr("src.web_app.session.st.warning", MagicMock())

    def test_first_render_sets_baseline_without_saving(self):
        from src.web_app.session import save_if_changed

        save = MagicMock()
        save_if_changed("k", {"example": True}, save, "x")
        save_if_changed("k", {"example": True}, save, "x")
        save.assert_not_called()

    def test_saves_once_per_real_change(self):
        from src.web_app.session import save_if_changed

        save = MagicMock()
        save_if_changed("k", [1], save, "x")
        save_if_changed("k", [1, 2], save, "x")
        save_if_changed("k", [1, 2], save, "x")
        save.assert_called_once_with([1, 2])

    def test_save_failure_warns_instead_of_raising(self):
        from src.web_app import session

        save_if_changed = session.save_if_changed
        failing = MagicMock(side_effect=RuntimeError("db down"))
        save_if_changed("k", 1, failing, "Couldn't save")
        save_if_changed("k", 2, failing, "Couldn't save")
        session.st.warning.assert_called_once_with("Couldn't save")
