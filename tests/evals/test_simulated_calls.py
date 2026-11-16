"""Simulated calls (design doc, Evaluation layer 5): whole calls over real audio.

Each scenario in tests/evals/sim_calls.py is played by a scripted caller
(macOS `say`, pauses, noise, accents, cut-ins, an LLM-written follow-up,
silence) against the real worker on the local LiveKit server, with real
Deepgram, Claude and ElevenLabs. Code checks what the call left behind; a
Haiku judge scores how natural the answers sounded.

Run with the voice venv, the local LiveKit server up and no other worker running:
    FINNIE_EVAL_CALLS=1 .venv-voice/bin/python -m pytest tests/evals/test_simulated_calls.py -v -s --no-cov
    FINNIE_EVAL_CALLS=1 FINNIE_EVAL_CALL_SCENARIOS=basic,interruption .venv-voice/bin/python -m pytest ...

Against LiveKit Cloud instead (adaptive interruption only works there: on the
dev server it falls back to VAD and `backchannel` fails), export LIVEKIT_URL,
LIVEKIT_API_KEY and LIVEKIT_API_SECRET in the shell first; the caller and the
worker both use them.

Cost: roughly 5-15 cents a scenario (speech, Claude, LiveKit minutes), about
$1.50 for all of them. Writes tests/evals/.sim_runs/report.json (git-ignored).
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from tests.evals.sim_calls import EVAL_WORKER_SETTINGS, Observed, by_name, run_checks

LIVE = os.environ.get("FINNIE_EVAL_CALLS", "0") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="Set FINNIE_EVAL_CALLS=1 (real calls; costs money)")
if LIVE:
    pytest.importorskip("livekit.rtc", reason="Run with the voice venv (.venv-voice)")

SELECTED = [n for n in os.environ.get("FINNIE_EVAL_CALL_SCENARIOS", "").split(",") if n]
SCENARIOS = by_name(SELECTED)
REPORT = Path(__file__).parent / ".sim_runs" / "report.json"
NATURALNESS_FLOOR = 3.5
_results: dict[str, dict[str, Any]] = {}


@pytest.fixture(scope="module")
def worker():
    from tests.evals.sim_caller import Worker

    w = Worker(EVAL_WORKER_SETTINGS)
    try:
        w.wait_ready()
        yield w
    finally:
        w.stop()
        _write_report(w)


@pytest.fixture
def real_keys(monkeypatch):
    """The judge and LLM caller need the real Anthropic key (tests/conftest.py fakes it)."""
    from tests.evals.sim_caller import _env_file

    key = _env_file().get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY missing from .env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    # The caller mints through the app's minter (spike/voice/local_token.py) in this process.
    monkeypatch.setenv("FINNIE_VOICE_ENABLED", "1")
    monkeypatch.setenv("FINNIE_VOICE_ALLOWED_EMAILS", "tester@example.com")
    # The local dev server unless the shell names another (LiveKit Cloud, for adaptive interruption).
    for name, dev in (("LIVEKIT_URL", "ws://localhost:7880"), ("LIVEKIT_API_KEY", "devkey"),
                      ("LIVEKIT_API_SECRET", "secret")):
        monkeypatch.setenv(name, os.environ.get(name, dev))
    from src.core.config import get_settings
    get_settings.cache_clear()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_scenario(scenario, worker, real_keys):
    from tests.evals.sim_caller import run_scenario

    observed = asyncio.run(run_scenario(scenario, worker))
    results = run_checks(scenario, observed)
    _results[scenario.name] = _record(observed, results)
    print(f"\n[{scenario.name}] said={observed.said} first_audio={observed.first_audio_s} "
          f"turns={[(t.get('agent'), t.get('question')) for t in observed.turns]}")
    failed = {name: why for name, why in results.items() if why}
    assert not failed, f"{scenario.name}: {failed}"


class Naturalness(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = sounds like a person talking, 1 = reads like a document")
    reason: str


_RUBRIC = (
    "You are judging an answer a voice assistant SPOKE aloud on a phone call, after the caller's "
    "question. Score 1-5 how natural it sounds when heard, not read: short sentences, no lists or "
    "headings read out, no symbols, a direct answer first, a length that suits the question."
)


def test_answers_sound_natural(real_keys):
    """Run last (file order): judges every answered turn the scenarios above produced."""
    turns = [(t.get("question") or "", t.get("spoken") or "")
             for r in _results.values() for t in r["turns"] if t.get("spoken")]
    if not turns:
        pytest.skip("no answered turns to judge (run the scenarios first)")
    from langchain_anthropic import ChatAnthropic

    from src.core.config import get_settings

    judge = ChatAnthropic(model=get_settings().voice.classifier_model,
                          anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
                          max_tokens=300).with_structured_output(Naturalness)
    scores = []
    for question, spoken in turns:
        verdict = judge.invoke([{"role": "system", "content": _RUBRIC},
                                {"role": "user", "content": f"Caller: {question}\n\nAssistant said: {spoken}"}])
        scores.append(verdict.score)
        print(f"  {verdict.score}  {question[:60]!r}: {verdict.reason}")
    _results["_naturalness"] = {"scores": scores, "mean": statistics.mean(scores)}
    assert statistics.mean(scores) >= NATURALNESS_FLOOR, scores


def _record(observed: Observed, results: dict[str, Any]) -> dict[str, Any]:
    return {
        "checks": results,
        "said": observed.said,
        "first_audio_s": observed.first_audio_s,
        "turns": [{k: t.get(k) for k in ("question", "agent", "preamble", "spoken", "interrupted")}
                  for t in observed.turns],
        "answers": [a.get("answer") for a in observed.answers],
        "summary": {k: observed.summary.get(k) for k in ("reason", "turns", "interruptions", "cost_usd", "seconds")},
        "ended_by_worker": observed.ended_by_worker,
    }


def _write_report(worker) -> None:
    if not _results:
        return
    first = [s for r in _results.values() for s in r.get("first_audio_s", []) if s is not None]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "settings": EVAL_WORKER_SETTINGS,
        "first_audio_p50": statistics.median(first) if first else None,
        "cost_usd": round(sum((r.get("summary") or {}).get("cost_usd") or 0 for r in _results.values()), 4),
        "scenarios": _results,
        "worker_log": str(worker.log),
    }, indent=2, default=str))
    print(f"\nreport: {REPORT}")
