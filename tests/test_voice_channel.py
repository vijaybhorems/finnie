"""channel="voice": short spoken answers, kept out of the shared FAQ cache."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agents.base_agent import _DISCREET_AMOUNTS, _SPOKEN_AMOUNTS, _VOICE_INSTRUCTION
from src.core.state import FinnieState, turn_state_reset
from tests.test_prompt_caching import _finance_agent, _state
from tests.test_workflow_fast_path import fast_path  # noqa: F401 (fixture)


def _voice_state(query: str = "What is a P/E ratio?") -> FinnieState:
    state = _state(query)
    state.channel = "voice"
    return state


class TestVoicePrompt:
    def test_instruction_goes_in_the_per_request_block_only(self):
        shared, role, context = _finance_agent()._build_messages(_voice_state(), "ctx")[0].content
        assert _VOICE_INSTRUCTION in context["text"]
        assert "cache_control" not in context
        # The cached prefix is byte-identical to a chat turn's.
        chat_shared, chat_role, _ = _finance_agent()._build_messages(_state(), "ctx")[0].content
        assert (shared, role) == (chat_shared, chat_role)

    def test_chat_turns_get_no_instruction(self):
        blocks = _finance_agent()._build_messages(_state(), "ctx")[0].content
        assert all(_VOICE_INSTRUCTION not in b["text"] for b in blocks)

    def test_instruction_sent_even_without_other_context(self):
        blocks = _finance_agent()._build_messages(_voice_state(), "")[0].content
        assert len(blocks) == 3 and _VOICE_INSTRUCTION in blocks[2]["text"]


class TestVoiceAndTheFAQCache:
    def _run(self, graph, query: str, channel: str) -> dict:
        return graph.invoke(FinnieState(messages=[HumanMessage(content=query)], channel=channel))

    def test_voice_answer_is_never_served_to_chat(self, fast_path):  # noqa: F811
        graph = fast_path["graph"]
        self._run(graph, "What is a P/E ratio?", "voice")
        result = self._run(graph, "What is a P/E ratio?", "chat")
        assert result["cache_hit"] is False
        assert fast_path["agent"].calls == 2

    def test_voice_does_not_read_chat_answers(self, fast_path):  # noqa: F811
        graph = fast_path["graph"]
        self._run(graph, "What is a P/E ratio?", "chat")
        result = self._run(graph, "What is a P/E ratio?", "voice")
        assert result["cache_hit"] is False
        assert fast_path["agent"].calls == 2

    def test_chat_still_caches(self, fast_path):  # noqa: F811
        graph = fast_path["graph"]
        self._run(graph, "What is a P/E ratio?", "chat")
        assert self._run(graph, "What is a P/E ratio?", "chat")["cache_hit"] is True


def test_channel_resets_to_chat_each_turn():
    assert turn_state_reset()["channel"] == "chat"


def test_persistent_turn_carries_the_channel():
    from src.workflow.graph import _prepare_turn

    _, turn, _, _ = _prepare_turn("hi", None, None, "g-1", "g-1:abc", "voice")
    assert turn["channel"] == "voice"


class TestInterruptedAfter:
    """What the user heard before cutting in reaches the next turn's prompt, and only that turn."""

    def test_note_goes_in_the_per_request_block_only(self):
        state = _voice_state("No, I meant Roth")
        state.interrupted_after = "A Roth IRA is funded with after-tax money, so"
        shared, role, context = _finance_agent()._build_messages(state, "ctx")[0].content
        assert "interrupted your last answer" in context["text"]
        assert '"A Roth IRA is funded with after-tax money, so"' in context["text"]
        chat_shared, chat_role, _ = _finance_agent()._build_messages(_state(), "ctx")[0].content
        assert (shared, role) == (chat_shared, chat_role)

    def test_long_heard_text_keeps_its_end(self):
        state = _voice_state()
        state.interrupted_after = "start " + "word " * 200 + "where they stopped"
        context = _finance_agent()._build_messages(state, "")[0].content[2]["text"]
        assert "where they stopped" in context and "start" not in context

    def test_no_note_without_an_interruption(self):
        blocks = _finance_agent()._build_messages(_voice_state(), "ctx")[0].content
        assert all("interrupted" not in b["text"] for b in blocks)

    def test_reset_each_turn_and_carried_by_the_turn_input(self):
        from src.workflow.graph import _prepare_turn

        assert turn_state_reset()["interrupted_after"] is None
        _, turn, _, _ = _prepare_turn("hi", None, None, "g-1", "g-1:abc", "voice", "heard this")
        assert turn["interrupted_after"] == "heard this"
        _, state, _, _ = _prepare_turn("hi", None, None, None, None, "voice", "heard this")
        assert state.interrupted_after == "heard this"


class TestSpeakAmounts:
    """Discreet by default (design doc, Open question 4); amounts only when the user turned them on."""

    def _context(self, speak: bool | None = None, channel: str = "voice") -> str:
        state = _voice_state() if channel == "voice" else _state()
        if speak is not None:
            state.speak_amounts = speak
        return "".join(b["text"] for b in _finance_agent()._build_messages(state, "ctx")[0].content)

    def test_voice_is_discreet_by_default(self):
        context = self._context()
        assert _DISCREET_AMOUNTS in context and _SPOKEN_AMOUNTS not in context

    def test_discreet_never_sends_the_caller_to_the_screen(self):
        """The page shows this same figure-free answer: "it's on your screen" would be false."""
        assert "Don't point them to their screen" in _DISCREET_AMOUNTS
        assert "say it's on their screen" not in _DISCREET_AMOUNTS

    def test_amounts_when_turned_on(self):
        context = self._context(speak=True)
        assert _SPOKEN_AMOUNTS in context and _DISCREET_AMOUNTS not in context

    def test_chat_gets_neither(self):
        context = self._context(speak=True, channel="chat")
        assert _SPOKEN_AMOUNTS not in context and _DISCREET_AMOUNTS not in context

    def test_rules_stay_out_of_the_cached_blocks(self):
        for speak in (False, True):
            state = _voice_state()
            state.speak_amounts = speak
            shared, role, _ = _finance_agent()._build_messages(state, "ctx")[0].content
            chat_shared, chat_role, _ = _finance_agent()._build_messages(_state(), "ctx")[0].content
            assert (shared, role) == (chat_shared, chat_role)

    def test_memories_are_used_silently(self):
        assert "don't recite it back unless they ask" in _VOICE_INSTRUCTION

    def test_reset_each_turn_and_carried_by_the_turn_input(self):
        from src.workflow.graph import _prepare_turn

        assert turn_state_reset()["speak_amounts"] is False
        _, turn, _, _ = _prepare_turn("hi", None, None, "g-1", "g-1:abc", "voice", None, True)
        assert turn["speak_amounts"] is True
        _, state, _, _ = _prepare_turn("hi", None, None, None, None, "voice", None, True)
        assert state.speak_amounts is True
