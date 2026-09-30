"""Live memory evals — real extraction model, real embeddings, real agent.

The unit tests (tests/test_memory.py) script the extractor; these check what
only the real model can show: that it extracts the right facts and skips the
excluded ones, that a restated fact replaces the old one, and that an answer in
a later session actually uses what was remembered. A few cents per run.

    FINNIE_EVAL_LIVE=1 pytest tests/evals/test_memory_evals.py -v -s
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import dotenv_values

LIVE_MODE = os.environ.get("FINNIE_EVAL_LIVE", "0") == "1"
_ROOT = Path(__file__).resolve().parents[2]


def _real_api_key() -> str:
    # tests/conftest.py replaces ANTHROPIC_API_KEY with a fake for every test.
    return dotenv_values(_ROOT / ".env").get("ANTHROPIC_API_KEY") or ""


pytestmark = [
    pytest.mark.skipif(not LIVE_MODE or not _real_api_key(), reason="Set FINNIE_EVAL_LIVE=1 with a key in .env"),
    pytest.mark.real_embeddings,
]


@pytest.fixture(autouse=True)
def live(monkeypatch):
    from src.core.config import get_settings
    from src.core.llm import get_llm
    from src.persistence.backend import reset_persistence
    from src.workflow.graph import build_graph

    monkeypatch.setenv("ANTHROPIC_API_KEY", _real_api_key())
    monkeypatch.setenv("DATABASE_URL", "")
    for cached in (get_settings, get_llm, build_graph):
        cached.cache_clear()
    reset_persistence()
    monkeypatch.setattr(get_settings().fast_path.faq_cache, "enabled", False)
    import src.workflow.graph as graph
    graph._AGENTS.clear()
    yield
    graph._AGENTS.clear()


def _user():
    from src.persistence.identity import user_id_from_claims
    from src.persistence.user_data import UserData

    return UserData(user_id_from_claims(str(uuid.uuid4().int)[:21], None))


def test_extracts_goals_situation_and_preferences():
    from src.memory.service import remember_turn

    user = _user()
    saved = remember_turn(
        user.user_id,
        "I'm 34, saving for a house deposit — aiming to buy in 2028. I prefer low-cost index funds.",
    )
    facts = " | ".join(saved).lower()
    print(f"\nsaved: {saved}")
    assert any(w in facts for w in ("house", "home")) and "2028" in facts
    assert "34" in facts
    assert "index" in facts


def test_skips_questions_and_hypotheticals():
    from src.memory.service import remember_turn

    saved = remember_turn(_user().user_id, "What if I retired at 50 — would I run out of money?")
    print(f"\nsaved: {saved}")
    assert saved == []


def test_never_stores_account_numbers():
    from src.memory.service import remember_turn

    user = _user()
    saved = remember_turn(user.user_id, "My brokerage account number is 88213456710 and I'm saving for retirement.")
    print(f"\nsaved: {saved}")
    assert not any(ch.isdigit() and "8821" in fact for fact in saved for ch in fact)
    assert all("88213456710" not in m["fact"] for m in user.list_memories())


def test_restated_fact_replaces_the_old_one():
    from src.memory.service import remember_turn

    user = _user()
    remember_turn(user.user_id, "I'm 34 years old.")
    remember_turn(user.user_id, "Actually I just turned 35 — I'm 35 years old now.")
    ages = [m["fact"] for m in user.list_memories() if "3" in m["fact"] and "old" in m["fact"].lower()]
    print(f"\nage memories: {ages}")
    assert len(ages) == 1 and "35" in ages[0]


def test_later_session_answer_uses_the_remembered_goal():
    """The design doc's exit criterion, end to end with the real agent."""
    from src.memory.service import remember_turn
    from src.workflow.graph import run_workflow

    user = _user()
    remember_turn(user.user_id, "I'm saving for a house deposit and plan to buy in 2028.")

    with patch("src.agents.finance_qa_agent.FredClient") as fred:
        fred.return_value.get_macro_snapshot.return_value = {}
        result = run_workflow(
            "How should I think about how much of my savings to keep in bonds?",
            user_id=user.user_id,
            thread_id=user.new_thread(),          # a new session: no chat history
        )
    answer = result["final_response"].lower()
    print(f"\nagent: {result['agent_used']}\nanswer (first 600 chars):\n{result['final_response'][:600]}")
    assert "2028" in answer or "house" in answer or "home" in answer
