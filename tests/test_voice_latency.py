"""scripts/voice_latency.py: p50/p95 per step and turn type from worker records."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "voice_latency", Path(__file__).resolve().parents[1] / "scripts" / "voice_latency.py",
)
vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vl)


def _turn(agent: str, first_audio: float, **extra) -> dict:
    return {"agent": agent, "time_to_first_audio_s": first_audio, "verdict_s": 0.0, **extra}


def test_reads_turn_logs_structured_logs_and_cloud_exports():
    lines = [
        json.dumps({"at": "2026-11-15T10:00:00", "question": "q", **_turn("finance_qa", 2.0)}),
        "2026-11-15 10:00:01 [info] registered worker",                     # not JSON
        json.dumps({"event": "voice_turn", "timestamp": "2026-11-15T10:01:00Z", **_turn("portfolio", 3.0)}),
        json.dumps({"event": "voice_call_ended", "seconds": 40}),           # another event
        json.dumps({"jsonPayload": {"event": "voice_turn", **_turn("tax_education", 2.5)}}),
        "{not json",
    ]
    assert [r["agent"] for r in vl.read_records(lines)] == ["finance_qa", "portfolio", "tax_education"]


def test_since_skips_older_records():
    lines = [json.dumps({"at": "2026-11-14T10:00:00", **_turn("finance_qa", 2.0)}),
             json.dumps({"at": "2026-11-15T10:00:00", **_turn("finance_qa", 4.0)})]
    assert [r["time_to_first_audio_s"] for r in vl.read_records(lines, since="2026-11-15")] == [4.0]


def test_percentiles_by_turn_type():
    records = [_turn("finance_qa", s) for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] + [_turn("goal_planning", 9.0)]
    summary = vl.summarise(records)
    assert summary["finance_qa"]["turns"] == 10
    assert summary["finance_qa"]["time_to_first_audio_s"] == (5.0, 10.0)
    assert summary["all"]["turns"] == 11
    assert summary["goal_planning"]["tts_ttfb_s"] is None  # not recorded


def test_render_and_main(tmp_path, capsys):
    log = tmp_path / "turns.jsonl"
    log.write_text("\n".join(json.dumps(_turn("finance_qa", s)) for s in (2.0, 3.0)))
    assert vl.main([str(log)]) == 0
    out = capsys.readouterr().out
    assert "| finance_qa | 2 |" in out and "2.00 / 3.00" in out
    assert vl.main([str(tmp_path / "turns.jsonl"), "--since", "2099"]) == 1
