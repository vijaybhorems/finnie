"""src/voice/usage.py: the voice_call_summary line, token counting, cost estimate."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableLambda

from src.core.config import TokenRates, VoiceCostConfig
from src.voice.usage import TokenMeter, call_summary, estimate_cost

RATES = VoiceCostConfig(
    livekit_per_minute=0.01,
    stt_per_minute=0.006,
    tts_per_1k_chars={"eleven_flash_v2_5": 0.05},
    llm_per_mtok={
        "claude-sonnet-5": TokenRates(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5),
        "claude-haiku-5-5": TokenRates(input=0.1, output=0.5, cache_read=0.01, cache_write=0.125),
    },
)


def _result(model: str, input_tokens: int, output: int, read: int = 0, created: int = 0) -> LLMResult:
    message = AIMessage(content="x", response_metadata={"model": model, "usage": {
        "input_tokens": input_tokens, "output_tokens": output,
        "cache_read_input_tokens": read, "cache_creation_input_tokens": created,
    }})
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestTokenMeter:
    def test_adds_up_by_model(self):
        meter = TokenMeter()
        meter.on_llm_end(_result("claude-sonnet-5", 100, 50, read=4000))
        meter.on_llm_end(_result("claude-sonnet-5", 20, 10, created=30))
        meter.on_llm_end(_result("claude-haiku-5-5", 300, 40))
        assert meter.by_model() == {
            "claude-sonnet-5": {"input_tokens": 120, "cache_read_input_tokens": 4000,
                                "cache_creation_input_tokens": 30, "output_tokens": 60},
            "claude-haiku-5-5": {"input_tokens": 300, "cache_read_input_tokens": 0,
                                 "cache_creation_input_tokens": 0, "output_tokens": 40},
        }

    def test_streamed_usage_metadata_counts_too(self):
        """Streamed answers may carry only LangChain's usage_metadata (input includes cache)."""
        message = AIMessage(content="x", response_metadata={"model_name": "claude-sonnet-5"}, usage_metadata={
            "input_tokens": 1100, "output_tokens": 7, "total_tokens": 1107,
            "input_token_details": {"cache_read": 1000},
        })
        meter = TokenMeter()
        meter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        assert meter.by_model()["claude-sonnet-5"]["input_tokens"] == 100
        assert meter.by_model()["claude-sonnet-5"]["cache_read_input_tokens"] == 1000

    def test_reaches_model_calls_nested_inside_a_run(self):
        """How the graph's agents are reached: they call their model without passing config."""
        model = GenericFakeChatModel(messages=iter([AIMessage(
            content="hi", response_metadata={"model": "claude-haiku-5-5"},
            usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        )]))
        node = RunnableLambda(lambda q: model.invoke(q).content)  # no config passed on
        meter = TokenMeter()
        assert node.invoke("q", config={"callbacks": [meter]}) == "hi"
        assert meter.by_model()["claude-haiku-5-5"]["input_tokens"] == 12

    def test_threads_do_not_lose_counts(self):
        meter = TokenMeter()
        threads = [threading.Thread(target=lambda: [meter.on_llm_end(_result("m", 1, 1)) for _ in range(500)])
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert meter.by_model()["m"]["input_tokens"] == 2000

    def test_responses_without_usage_are_skipped(self):
        meter = TokenMeter()
        meter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]]))
        assert meter.by_model() == {}


class TestEstimateCost:
    def test_prices_each_item(self):
        tokens = {"claude-sonnet-5": {"input_tokens": 1_000_000, "output_tokens": 100_000,
                                      "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 0}}
        cost = estimate_cost(seconds=61, stt_seconds=60, tts_chars={"eleven_flash_v2_5": 2000},
                             tokens=tokens, rates=RATES)
        assert cost["cost_by_item"] == {
            "livekit": 0.02,   # 61 s is two started minutes
            "stt": 0.006,
            "tts": 0.1,
            "llm": 3.2,        # 2.0 input + 1.0 output + 0.2 cache reads
        }
        assert cost["cost_usd"] == pytest.approx(3.326)
        assert cost["unpriced"] == []

    def test_dated_snapshot_names_match_but_other_models_do_not(self):
        tokens = {
            "claude-sonnet-5-20260801": {"input_tokens": 1_000_000},
            "claude-sonnet-5-5": {"input_tokens": 1_000_000},  # prefixed by claude-sonnet-5, priced differently
        }
        cost = estimate_cost(0, 0, {}, tokens, RATES)
        assert cost["cost_by_item"]["llm"] == 2.0
        assert cost["unpriced"] == ["claude-sonnet-5-5"]

    def test_unknown_voice_is_listed_not_guessed(self):
        cost = estimate_cost(0, 0, {"aura-2-asteria-en": 500, "silent": 0}, {}, RATES)
        assert cost["cost_by_item"]["tts"] == 0
        assert cost["unpriced"] == ["aura-2-asteria-en"]


class TestCallSummary:
    def _summary(self, **overrides):
        args = dict(
            user_id="g-123", reason="idle_hangup", seconds=95.4,
            turns=[{"agent": "finance_qa", "question": "secret words", "spoken": "more words"},
                   {"agent": "finance_qa"}, {"agent": "error"}],
            interruptions=1, stt_seconds=None, tts_chars={"eleven_flash_v2_5": 800},
            tokens={}, vendor_failures={"elevenlabs"}, backup_voice_used=True, rates=RATES,
        )
        return call_summary(**{**args, **overrides})

    def test_fields(self):
        s = self._summary()
        assert s["reason"] == "idle_hangup"
        assert s["turns"] == 3 and s["unanswered_turns"] == 1
        assert s["agents"] == {"finance_qa": 2, "error": 1}
        assert s["interruptions"] == 1 and s["backup_voice_used"] is True
        assert s["vendor_failures"] == ["elevenlabs"]
        assert s["stt_seconds"] == 95.4  # no STT usage reported: the mic streamed the whole call

    def test_never_carries_what_was_said(self):
        assert "secret words" not in repr(self._summary())
        assert "more words" not in repr(self._summary())

    def test_reported_stt_audio_wins(self):
        assert self._summary(stt_seconds=40.0)["stt_seconds"] == 40.0


def test_classifier_passes_callbacks_to_its_model_call(monkeypatch):
    from src.workflow import classify

    structured = MagicMock()
    structured.invoke.return_value = classify.Verdict(on_topic=True, agent="finance_qa", reason="r")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(classify, "get_llm", lambda **kw: llm)

    meter = TokenMeter()
    classify.classify_query("what is an index fund", [meter])
    assert structured.invoke.call_args.kwargs["config"] == {"callbacks": [meter]}

    classify.clear_verdict_cache()
    classify.classify_query("what is an index fund")
    assert structured.invoke.call_args.kwargs["config"] is None


def test_every_run_is_tagged_with_its_channel():
    from src.workflow.graph import _prepare_turn

    meter = TokenMeter()
    _, _, config, _ = _prepare_turn("hi", None, None, None, None, "voice", callbacks=[meter])
    assert config["metadata"] == {"channel": "voice"}
    assert config["tags"] == ["channel:voice"]
    assert config["callbacks"] == [meter]

    _, _, config, _ = _prepare_turn("hi", None, None, None, None)
    assert config["metadata"] == {"channel": "chat"} and "callbacks" not in config
