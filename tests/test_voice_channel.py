"""channel="voice": short spoken answers, kept out of the shared FAQ cache."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agents.base_agent import _VOICE_INSTRUCTION
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
