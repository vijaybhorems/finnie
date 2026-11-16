"""The voice worker's turn, in LiveKit's text-only test mode: no audio, no room.

Needs the voice venv (requirements-voice.txt); skipped elsewhere, CI included:
    .venv-voice/bin/python -m pytest tests/test_voice_worker.py --no-cov

The graph, classifier and user settings are stubbed. What runs for real is
FinnieVoiceAgent.llm_node inside a LiveKit AgentSession: the preamble, the
speech shaping, the discreet-amount filter, the turn limit and memory saves.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents import AgentSession  # noqa: E402

from src.voice import worker  # noqa: E402
from src.voice.speech import AMOUNT_ON_SCREEN  # noqa: E402
from src.voice.turns import SECURITIES_LINE  # noqa: E402

USER, THREAD = "g-1", "g-1:" + "a" * 32


class FakeSaves:
    def __init__(self) -> None:
        self.added: list[tuple[str, str, float]] = []

    def add(self, user_id: str, message: str, min_confidence: float) -> None:
        self.added.append((user_id, message, min_confidence))

    def wait(self, timeout_s: float) -> int:
        return 0


@pytest.fixture
def graph(monkeypatch):
    """Stub the graph and classifier; record what each turn was given."""
    state: dict[str, Any] = {"answer": "A P/E ratio compares price to earnings.", "agent": "finance_qa",
                             "verdict_agent": "finance_qa", "tickers": [], "speak": False, "calls": []}

    def stream_workflow(question, sink=None, **kwargs):
        state["calls"].append({"question": question, **kwargs})
        sink["agent_used"] = state["agent"]
        yield from state["answer"].split(" ")[:1]
        for word in state["answer"].split(" ")[1:]:
            yield " " + word

    def classify_query(question):
        return SimpleNamespace(on_topic=True, agent=state["verdict_agent"], tickers=state["tickers"])

    monkeypatch.setattr("src.workflow.graph.stream_workflow", stream_workflow)
    monkeypatch.setattr("src.workflow.classify.classify_query", classify_query)
    monkeypatch.setattr(worker, "_speak_amounts", lambda *_: state["speak"])
    monkeypatch.setattr(worker, "FILLER", False)
    monkeypatch.setattr(worker, "SPECULATE", False)
    return state


async def _say(text: str, agent: worker.FinnieVoiceAgent) -> str:
    async with AgentSession(llm=worker.GraphLLM()) as session:
        await session.start(agent)
        result = await session.run(user_input=text)
        message = result.expect.next_event().is_message(role="assistant")
        return message.event().item.text_content


def _agent() -> worker.FinnieVoiceAgent:
    agent = worker.FinnieVoiceAgent(USER, THREAD)
    agent.memories = FakeSaves()
    return agent


async def test_a_turn_runs_the_graph_on_the_voice_channel(graph):
    agent = _agent()
    said = await _say("what is a p e ratio", agent)
    assert "A P E ratio compares price to earnings." in said
    call = graph["calls"][0]
    assert call["channel"] == "voice" and call["thread_id"] == THREAD and call["user_id"] == USER
    assert call["speak_amounts"] is False and call["interrupted_after"] is None


async def test_spelled_out_terms_reach_the_graph_written(graph):
    await _say("can I roll my four zero one k into an ira", _agent())
    assert graph["calls"][0]["question"] == "can I roll my 401(k) into an ira"


async def test_securities_line_comes_before_the_answer(graph):
    graph.update(answer="Apple is up 2% today.", agent="market_analysis", verdict_agent="market_analysis",
                 tickers=["AAPL"])
    said = await _say("how is apple doing", _agent())
    assert said.index(SECURITIES_LINE) < said.index("Apple is up")


async def test_amounts_are_kept_off_the_audio_unless_turned_on(graph):
    graph.update(answer="Your portfolio is worth $48,250, up 6%.", agent="portfolio", verdict_agent="portfolio")
    assert AMOUNT_ON_SCREEN in await _say("how is my portfolio", _agent())
    graph["speak"] = True
    said = await _say("how is my portfolio", _agent())
    assert "$48,250" in said and graph["calls"][-1]["speak_amounts"] is True


async def test_answered_turns_save_memories_refusals_do_not(graph):
    import asyncio

    agent = _agent()
    await _say("I'm saving for a house in 2028", agent)
    graph["agent"] = "out_of_scope"
    await _say("tell me a joke about cats", agent)
    await asyncio.sleep(0.1)  # saves are added from the graph's thread as it finishes
    assert agent.memories.added == [(USER, "I'm saving for a house in 2028", 0.9)]


async def test_the_turn_limit_says_goodbye_instead_of_answering(graph, monkeypatch):
    from src.voice.limits import TURN_LIMIT

    agent = _agent()
    agent.monitor = SimpleNamespace(turn_limit_reached=False, start_turn=lambda: False)
    assert TURN_LIMIT in await _say("one more question", agent)
    assert graph["calls"] == []
