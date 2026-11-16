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
from src.voice.turns import SECURITIES_LINE, SORRY  # noqa: E402

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
                             "verdict_agent": "finance_qa", "tickers": [], "speak": False, "calls": [],
                             "classified": []}

    def stream_workflow(question, sink=None, **kwargs):
        state["calls"].append({"question": question, **kwargs})
        sink["agent_used"] = state["agent"]
        yield from state["answer"].split(" ")[:1]
        for word in state["answer"].split(" ")[1:]:
            yield " " + word
        sink["final_response"] = state["answer"]  # as stream_workflow fills it, for the page

    def classify_query(question, callbacks=None, context=""):
        state["classified"].append((question, context))
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


async def test_every_turn_counts_its_tokens_on_the_calls_meter(graph):
    """The call summary's tokens and cost come from this one meter per call."""
    agent = _agent()
    await _say("what is a p e ratio", agent)
    await _say("and a dividend", agent)
    assert [c["callbacks"] for c in graph["calls"]] == [[agent.tokens], [agent.tokens]]


async def test_the_classifier_gets_the_previous_turn_as_the_graph_will(graph):
    """The early verdict is reused only if its context matches the graph's (classify.previous_turn)."""
    from src.workflow.classify import conversation_context

    agent = _agent()
    await _say("if i invest ten thousand dollars at seven percent how much will i have", agent)
    graph["answer"] = "About nineteen thousand seven hundred."
    await _say("whats the exact dollar amount", agent)
    assert graph["classified"][0] == ("if i invest ten thousand dollars at seven percent how much will i have", "")
    assert graph["classified"][1] == ("whats the exact dollar amount", conversation_context(
        "if i invest ten thousand dollars at seven percent how much will i have",
        "A P/E ratio compares price to earnings."))  # the first turn's final_response


async def test_stop_skips_the_classifier_and_reaches_the_graph(graph):
    graph.update(answer="Okay. Ask me whenever you're ready.", agent="acknowledge")
    agent = _agent()
    said = await _say("stop", agent)
    assert graph["classified"] == []
    assert graph["calls"][0]["question"] == "stop"
    assert "Okay." in said
    assert agent.memories.added == []


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


async def test_each_answer_reaches_the_page_written_in_full(graph, monkeypatch):
    """The page shows the full written answer, amounts included, numbered by turn."""
    import asyncio

    sent: list[tuple[str, dict]] = []

    def page_sender(user_id):
        async def send(event):
            sent.append((user_id, event))
        return send

    monkeypatch.setattr(worker, "page_sender", page_sender)
    graph.update(answer="Your portfolio is worth $48,250.", agent="portfolio", verdict_agent="portfolio")
    agent = _agent()
    await _say("how is my portfolio", agent)
    graph["agent"] = "error"
    await _say("and now", agent)
    await asyncio.sleep(0.1)  # sent from the graph's thread as each run finishes

    assert sent == [
        (USER, {"type": "answer", "turn": 1, "agent": "portfolio", "answer": "Your portfolio is worth $48,250."}),
        (USER, {"type": "answer", "turn": 2, "agent": "error", "answer": SORRY}),
    ]


async def test_the_graph_starts_while_the_verdict_is_still_coming(graph, monkeypatch):
    """The graph loads the thread and user data during the classifier call, not after it."""
    import threading

    graph_started = threading.Event()
    from src.workflow import graph as graph_module

    stub = graph_module.stream_workflow

    def stream_workflow(question, sink=None, **kwargs):
        graph_started.set()
        yield from stub(question, sink=sink, **kwargs)

    def classify_query(question, callbacks=None, context=""):
        graph["classified"].append((question, graph_started.wait(timeout=5)))
        return SimpleNamespace(on_topic=True, agent="finance_qa", tickers=[])

    monkeypatch.setattr("src.workflow.graph.stream_workflow", stream_workflow)
    monkeypatch.setattr("src.workflow.classify.classify_query", classify_query)
    said = await _say("what is a p e ratio", _agent())
    assert graph["classified"] == [("what is a p e ratio", True)]
    assert "A P E ratio compares price to earnings." in said


async def _items(agent: worker.FinnieVoiceAgent, text: str) -> list[Any]:
    from livekit.agents import ModelSettings, llm

    ctx = llm.ChatContext()
    ctx.add_message(role="user", content=text)
    return [item async for item in agent.llm_node(ctx, [], ModelSettings())]


async def test_the_first_sentence_goes_to_speech_at_once(graph):
    """Flushed after the first sentence: the sentence splitter would hold it until the next began."""
    from livekit.agents.types import FlushSentinel

    graph["answer"] = "A P/E ratio compares price to earnings. A high one means investors expect growth. More?"
    items = await _items(_agent(), "what is a p e ratio")
    flushes = [i for i, item in enumerate(items) if isinstance(item, FlushSentinel)]
    assert len(flushes) == 1
    assert "".join(items[:flushes[0]]).strip() == "A P E ratio compares price to earnings."


async def test_the_filler_is_flushed_before_the_answer(graph, monkeypatch):
    from livekit.agents.types import FlushSentinel

    from src.voice.prefetch import FILLERS

    monkeypatch.setattr(worker, "FILLER", True)
    graph.update(answer="Starting at age 30 helps. A lot.", agent="goal_planning", verdict_agent="goal_planning")
    items = await _items(_agent(), "how much should i save to retire")
    assert items[0] == FILLERS["goal_planning"] and isinstance(items[1], FlushSentinel)


async def test_a_final_transcript_is_classified_at_once(graph, monkeypatch):
    """Not queued behind an interim's call: the final wording exists only at the end."""
    import asyncio
    import threading

    release = threading.Event()

    def classify_query(question, callbacks=None, context=""):
        graph["classified"].append(question)
        if question == "what is a p":
            release.wait(timeout=5)  # an interim's call still running
        return SimpleNamespace(on_topic=True, agent="finance_qa", tickers=[])

    monkeypatch.setattr("src.workflow.classify.classify_query", classify_query)
    monkeypatch.setattr(worker, "SPECULATE", True)
    agent = _agent()
    agent.on_transcript("what is a p", is_final=False)
    await asyncio.sleep(0.05)  # its classification is under way
    agent.on_transcript("what is a pe ratio", is_final=False)  # queued behind the first
    agent.on_transcript("what is a pe ratio", is_final=True)   # starts now
    for _ in range(50):
        await asyncio.sleep(0.02)
        if "what is a pe ratio" in graph["classified"]:
            break
    assert graph["classified"] == ["what is a p", "what is a pe ratio"]
    release.set()
    await asyncio.sleep(0.1)
    # The queued interim has the same words and context: not classified twice.
    assert graph["classified"] == ["what is a p", "what is a pe ratio"]
