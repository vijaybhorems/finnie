"""What one voice call used and roughly cost: the `voice_call_summary` log line.

Design doc, Monitoring: calls and why each ended, call minutes per user per day,
interruptions per call, switches to the backup voice, provider errors and cost
per call all come from this one line per call (time to first audio comes from
the per-turn `voice_turn` lines). Like every worker log, it carries ids, counts
and timings, never what was said.

Claude tokens are counted by `TokenMeter`, a LangChain callback the worker
passes to each graph run and classifier call, since LiveKit can't see inside
the graph. Speech usage comes from LiveKit's own per-model usage, passed in as
plain numbers so this module stays importable without LiveKit.

The cost is an estimate from list prices in `voice.cost` (config.yaml): good
for spotting a call that costs ten times the others or a daily total heading
for the ceiling, not for reconciling invoices.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from typing import Any, Iterable, Optional

from langchain_core.callbacks import BaseCallbackHandler

from src.core.config import TokenRates, VoiceCostConfig
from src.core.llm import token_usage

TOKEN_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")


class TokenMeter(BaseCallbackHandler):
    """Claude tokens by model, from every model call made under it.

    Pass it as a callback: LangChain hands it to nested calls too, so one meter
    on a graph run counts the classifier and the agent. Thread-safe: the graph
    runs beside the worker's event loop.
    """

    raise_error = False  # a counting bug must never fail a turn

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, dict[str, int]] = {}

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        hint = (getattr(response, "llm_output", None) or {}).get("model_name", "")
        for generations in getattr(response, "generations", None) or []:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = token_usage(message) if message is not None else {}
                if not usage:
                    continue
                meta = getattr(message, "response_metadata", None) or {}
                model = meta.get("model_name") or meta.get("model") or hint or "unknown"
                with self._lock:
                    totals = self._by_model.setdefault(model, dict.fromkeys(TOKEN_FIELDS, 0))
                    for name in TOKEN_FIELDS:
                        totals[name] += int(usage.get(name, 0))

    def by_model(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {model: dict(totals) for model, totals in self._by_model.items()}


_DATED = re.compile(r"-\d{8}$")


def _rates_for(model: str, table: dict[str, TokenRates]) -> Optional[TokenRates]:
    """The configured rates for `model`, also under its dated snapshot name.

    Exact match after dropping a "-YYYYMMDD" suffix only: "claude-sonnet-5" is a
    prefix of "claude-sonnet-5-5", a different model with different prices.
    """
    return table.get(model) or table.get(_DATED.sub("", model))


def _llm_cost(tokens: dict[str, dict[str, int]], rates: VoiceCostConfig, unpriced: list[str]) -> float:
    total = 0.0
    for model, t in tokens.items():
        r = _rates_for(model, rates.llm_per_mtok)
        if r is None:
            unpriced.append(model)
            continue
        total += (
            t.get("input_tokens", 0) * r.input
            + t.get("output_tokens", 0) * r.output
            + t.get("cache_read_input_tokens", 0) * r.cache_read
            + t.get("cache_creation_input_tokens", 0) * r.cache_write
        ) / 1_000_000
    return total


def estimate_cost(
    seconds: float,
    stt_seconds: float,
    tts_chars: dict[str, int],
    tokens: dict[str, dict[str, int]],
    rates: VoiceCostConfig,
) -> dict[str, Any]:
    """Dollars by item and in total; models with no configured price are listed."""
    unpriced: list[str] = []
    tts = 0.0
    for model, chars in tts_chars.items():
        rate = rates.tts_per_1k_chars.get(model)
        if rate is None:
            if chars:
                unpriced.append(model)
            continue
        tts += chars / 1000 * rate
    items = {
        "livekit": math.ceil(max(seconds, 0) / 60) * rates.livekit_per_minute,
        "stt": max(stt_seconds, 0) / 60 * rates.stt_per_minute,
        "tts": tts,
        "llm": _llm_cost(tokens, rates, unpriced),
    }
    return {
        "cost_usd": round(sum(items.values()), 4),
        "cost_by_item": {name: round(value, 4) for name, value in items.items()},
        "unpriced": sorted(set(unpriced)),
    }


def call_summary(
    *,
    user_id: str,
    reason: str,
    seconds: float,
    turns: Iterable[dict[str, Any]],
    interruptions: int,
    stt_seconds: Optional[float],
    tts_chars: dict[str, int],
    tokens: dict[str, dict[str, int]],
    vendor_failures: Iterable[str],
    backup_voice_used: bool,
    rates: VoiceCostConfig,
) -> dict[str, Any]:
    """The fields of one call's `voice_call_summary` log line.

    `stt_seconds` is the audio Deepgram was sent; None (no usage reported) falls
    back to the call's length, since the microphone streams for the whole call.
    """
    turns = list(turns)
    stt = seconds if stt_seconds is None else stt_seconds
    return {
        "user_id": user_id,  # for minutes per user per day; an id, never an email
        "reason": reason,
        "seconds": round(seconds, 1),
        "turns": len(turns),
        "agents": dict(Counter(str(t.get("agent")) for t in turns)),
        "unanswered_turns": sum(1 for t in turns if t.get("agent") in (None, "error")),
        "interruptions": interruptions,
        "backup_voice_used": backup_voice_used,
        "vendor_failures": sorted(set(vendor_failures)),
        "stt_seconds": round(stt, 1),
        "tts_chars": dict(tts_chars),
        "tokens": tokens,
        **estimate_cost(seconds, stt, tts_chars, tokens, rates),
    }
