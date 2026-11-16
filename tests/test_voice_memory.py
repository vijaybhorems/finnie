"""src/voice/memory.py: which voice turns save memories, and waiting for saves at call end."""
from __future__ import annotations

import threading
from concurrent.futures import Future

import pytest

from src.core.config import VoiceConfig
from src.voice.memory import MemorySaves, should_remember

VOICE = VoiceConfig(memory_min_transcript_confidence=0.85)


class TestShouldRemember:
    @pytest.mark.parametrize("agent", ["finance_qa", "portfolio", "goal_planning"])
    def test_answered_turns_heard_clearly(self, agent):
        assert should_remember(agent, 0.97, VOICE)

    @pytest.mark.parametrize("agent", ["out_of_scope", "error", None, ""])
    def test_refused_failed_or_unknown_turns_are_skipped(self, agent):
        assert not should_remember(agent, 0.99, VOICE)

    def test_a_transcript_deepgram_was_unsure_of_is_skipped(self):
        assert not should_remember("finance_qa", 0.6, VOICE)
        assert should_remember("finance_qa", 0.85, VOICE)

    def test_no_confidence_given_is_not_held_against_the_turn(self):
        assert should_remember("finance_qa", None, VOICE)


class TestMemorySaves:
    def test_passes_the_voice_confidence_bar(self):
        calls = []

        def schedule(user_id, message, *, min_confidence):
            calls.append((user_id, message, min_confidence))
            future: Future = Future()
            future.set_result([])
            return future

        saves = MemorySaves(schedule)
        saves.add("g-1", "I'm saving for a house", 0.9)
        assert calls == [("g-1", "I'm saving for a house", 0.9)]
        assert saves.wait(1) == 0

    def test_waits_for_saves_still_running(self):
        release = threading.Event()
        done = []

        def slow(*_, **__):
            future: Future = Future()

            def finish():
                release.wait(5)
                done.append(True)
                future.set_result([])

            threading.Thread(target=finish).start()
            return future

        saves = MemorySaves(slow)
        saves.add("g-1", "I'm 34", 0.9)
        threading.Timer(0.05, release.set).start()
        assert saves.wait(5) == 0 and done == [True]

    def test_a_save_that_misses_the_wait_is_counted_not_retried(self):
        never: Future = Future()
        saves = MemorySaves(lambda *_, **__: never)
        saves.add("g-1", "I'm 34", 0.9)
        assert saves.wait(0.05) == 1
        assert saves.wait(0.05) == 0  # already given up on


class TestVoiceConfidenceBar:
    """remember_turn(min_confidence=...) raises the bar for voice; it never lowers chat's."""

    def _remember(self, monkeypatch, confidence, min_confidence):
        from src.memory import service
        from src.memory.extractor import ExtractedFact

        monkeypatch.setattr(service, "extract_facts", lambda _m: [
            ExtractedFact(fact="Saving for a house.", category="goal", confidence=confidence),
        ])
        monkeypatch.setattr(service, "UserData", _FakeUserData)
        return service.remember_turn("g-1", "I'm saving for a house", min_confidence=min_confidence)

    def test_voice_bar_drops_what_chat_would_keep(self, monkeypatch):
        assert self._remember(monkeypatch, 0.85, None) == ["Saving for a house."]
        assert self._remember(monkeypatch, 0.85, 0.9) == []

    def test_a_lower_bar_never_loosens_chats(self, monkeypatch):
        assert self._remember(monkeypatch, 0.7, 0.5) == []


class _FakeUserData:
    def __init__(self, *_):
        pass

    def memory_enabled(self):
        return True

    def remember(self, *_):
        pass
