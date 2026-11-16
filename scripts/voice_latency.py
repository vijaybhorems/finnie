"""Voice latency by step and turn type, from the worker's per-turn records.

Design doc, Evaluation layer 4: free, from logs. Reads either source the
worker writes:
- the local turn log (VOICE_TURN_LOG, spike/voice/runs/turns.jsonl), or
- its structured logs: lines whose event is "voice_turn", also inside a Cloud
  Logging export's "jsonPayload". These carry timings only, never words.
Other lines are skipped, so a whole worker log can be passed as it is.

    python scripts/voice_latency.py spike/voice/runs/turns.jsonl
    python scripts/voice_latency.py worker.log --since 2026-11-15
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

# What the user waits through, in order (seconds).
STEPS = [
    ("end_of_turn_delay_s", "turn decision"),      # end of speech -> LiveKit decides the turn ended
    ("transcription_delay_s", "final transcript"),  # end of speech -> final transcript
    ("verdict_s", "classifier"),                    # ~0 when classified while the user spoke
    ("finnie_first_token_s", "first token"),        # graph start -> first answer token
    ("llm_node_ttfs_s", "first sentence"),          # turn start -> first sentence sent to speech
    ("tts_ttfb_s", "speech first byte"),            # text-to-speech first audio
    ("time_to_first_audio_s", "first audio"),       # end of speech -> first audio, all of it
]


def read_records(lines: Iterable[str], since: str = "") -> list[dict[str, Any]]:
    records = []
    for line in lines:
        start = line.find("{")
        if start < 0:
            continue
        try:
            obj = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        obj = obj.get("jsonPayload", obj)
        is_turn_log = "time_to_first_audio_s" in obj and "event" not in obj
        if obj.get("event") != "voice_turn" and not is_turn_log:
            continue
        when = str(obj.get("at") or obj.get("timestamp") or "")
        if since and (not when or when < since):
            continue
        records.append(obj)
    return records


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def summarise(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{turn type: {"turns": n, step: (p50, p95)}}; "all" first, then by agent."""
    groups: dict[str, list[dict[str, Any]]] = {"all": records}
    for record in records:
        groups.setdefault(str(record.get("agent") or "unknown"), []).append(record)
    summary = {}
    for name, rows in groups.items():
        entry: dict[str, Any] = {"turns": len(rows)}
        for key, _ in STEPS:
            values = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
            entry[key] = (percentile(values, 50), percentile(values, 95)) if values else None
        summary[name] = entry
    return summary


def render(summary: dict[str, dict[str, Any]]) -> str:
    head = "| turn type | turns | " + " | ".join(label for _, label in STEPS) + " |"
    rule = "|---|---:|" + "---:|" * len(STEPS)
    rows = [head, rule]
    for name, entry in summary.items():
        cells = [
            f"{entry[key][0]:.2f} / {entry[key][1]:.2f}" if entry[key] else "–" for key, _ in STEPS
        ]
        rows.append(f"| {name} | {entry['turns']} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n\nSeconds, p50 / p95."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--since", default="", help="ISO date or time; earlier records are skipped")
    args = parser.parse_args(argv)
    records = []
    for path in args.files:
        records += read_records(path.read_text().splitlines(), args.since)
    if not records:
        print("no voice turns found", file=sys.stderr)
        return 1
    print(render(summarise(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
