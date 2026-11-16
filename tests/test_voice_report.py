"""scripts/voice_report.py: calls by day from voice_call_summary lines."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vr = _load("voice_report")
vl = _load("voice_latency")


def _call(day: str, user: str, seconds: float, **extra) -> dict:
    return {"event": "voice_call_summary", "timestamp": f"{day}T10:00:00Z", "user_id": user,
            "seconds": seconds, "reason": "user_initiated", "turns": 4, "interruptions": 1,
            "unanswered_turns": 0, "backup_voice_used": False, "vendor_failures": [],
            "cost_usd": 0.05, "unpriced": [], **extra}


CALLS = [
    _call("2026-11-15", "g-a", 120, reason="idle_hangup"),
    _call("2026-11-15", "g-a", 240, interruptions=3, vendor_failures=["elevenlabs"], backup_voice_used=True),
    _call("2026-11-15", "g-b", 60, cost_usd=0.5, unanswered_turns=1, unpriced=["claude-sonnet-5-5"]),
    _call("2026-11-16", "g-b", 30),
]


def test_reads_worker_logs_jsonl_and_gcloud_arrays():
    jsonl = "\n".join([
        "2026-11-15 10:00:01 [info] registered worker",
        json.dumps(CALLS[0]),
        json.dumps({"event": "voice_turn", "time_to_first_audio_s": 3}),
        json.dumps({"jsonPayload": CALLS[1], "timestamp": "2026-11-15T10:05:00Z"}),
    ])
    assert len(vr.read_summaries(jsonl)) == 2
    array = json.dumps([{"jsonPayload": c, "timestamp": c["timestamp"]} for c in CALLS], indent=2)
    assert len(vr.read_summaries(array)) == 4


def test_figures_by_day():
    day = vr.summarise(CALLS)["2026-11-15"]
    assert day["calls"] == 3 and day["users"] == 2
    assert day["minutes"] == 7.0 and day["most_minutes_by_one_user"] == 6.0  # g-a: 2 + 4 minutes
    assert day["ended"] == {"user_initiated": 2, "idle_hangup": 1}
    assert day["interruptions_per_call"] == 1.67
    assert day["backup_voice_calls"] == 1 and day["vendor_failures"] == {"elevenlabs": 1}
    assert day["unanswered_turns"] == 1
    assert day["cost_usd"] == 0.6 and day["max_call_cost_usd"] == 0.5
    assert day["cost_per_minute_usd"] == round(0.6 / 7, 3)
    assert day["unpriced"] == ["claude-sonnet-5-5"]
    assert list(vr.summarise(CALLS)) == ["2026-11-15", "2026-11-16"]


def test_cli(tmp_path, capsys):
    path = tmp_path / "calls.json"
    path.write_text(json.dumps([{"jsonPayload": c, "timestamp": c["timestamp"]} for c in CALLS]))
    assert vr.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "| 2026-11-15 | 3 | 2 | 7.0 |" in out and "unpriced" in out
    (tmp_path / "empty.log").write_text("nothing here\n")
    assert vr.main([str(tmp_path / "empty.log")]) == 1


def test_latency_script_reads_gcloud_arrays_too(tmp_path, capsys):
    turn = {"event": "voice_turn", "agent": "finance_qa", "time_to_first_audio_s": 2.5}
    path = tmp_path / "turns.json"
    path.write_text(json.dumps([{"jsonPayload": turn, "timestamp": "2026-11-15T10:00:00Z"}], indent=2))
    assert vl.main([str(path)]) == 0
    assert "| finance_qa | 1 |" in capsys.readouterr().out
