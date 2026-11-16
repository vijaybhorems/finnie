"""Voice calls by day, from the worker's `voice_call_summary` log lines.

Design doc, Monitoring: calls and why each ended, call minutes per user per day,
interruptions per call, switches to the backup voice, provider errors and cost
per call. (Time to first audio: scripts/voice_latency.py.) Reads a worker log
as it is, or a Cloud Logging export, either JSON lines or `gcloud logging read
--format=json`'s array:

    gcloud logging read 'jsonPayload.event="voice_call_summary"' --freshness=14d \\
        --format=json > calls.json
    python scripts/voice_report.py calls.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

EVENT = "voice_call_summary"


def read_summaries(text: str) -> list[dict[str, Any]]:
    """Every voice_call_summary record in `text`; other lines are skipped."""
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            entries: Iterable[Any] = json.loads(stripped)
        except json.JSONDecodeError:
            entries = []
    else:
        entries = []
        for line in text.splitlines():
            start = line.find("{")
            if start < 0:
                continue
            try:
                entries.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                continue
    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        payload = entry.get("jsonPayload", entry)
        if payload.get("event") != EVENT:
            continue
        payload = dict(payload)
        payload.setdefault("timestamp", entry.get("timestamp", ""))
        records.append(payload)
    return records


def summarise(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{day: figures}, oldest day first."""
    days: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        days[str(record.get("timestamp", ""))[:10] or "unknown"].append(record)
    summary = {}
    for day in sorted(days):
        rows = days[day]
        minutes_by_user: Counter = Counter()
        for r in rows:
            minutes_by_user[r.get("user_id", "?")] += float(r.get("seconds") or 0) / 60
        costs = [float(r["cost_usd"]) for r in rows if isinstance(r.get("cost_usd"), (int, float))]
        minutes = sum(minutes_by_user.values())
        vendor_failures: Counter = Counter()
        for r in rows:
            vendor_failures.update(r.get("vendor_failures") or [])
        summary[day] = {
            "calls": len(rows),
            "users": len(minutes_by_user),
            "ended": dict(Counter(str(r.get("reason")) for r in rows).most_common()),
            "minutes": round(minutes, 1),
            "most_minutes_by_one_user": round(max(minutes_by_user.values(), default=0), 1),
            "turns_per_call": round(statistics.mean(int(r.get("turns") or 0) for r in rows), 1),
            "interruptions_per_call": round(statistics.mean(int(r.get("interruptions") or 0) for r in rows), 2),
            "unanswered_turns": sum(int(r.get("unanswered_turns") or 0) for r in rows),
            "backup_voice_calls": sum(1 for r in rows if r.get("backup_voice_used")),
            "vendor_failures": dict(vendor_failures),
            "cost_usd": round(sum(costs), 2),
            "cost_per_call_usd": round(statistics.mean(costs), 3) if costs else None,
            "max_call_cost_usd": round(max(costs), 3) if costs else None,
            "cost_per_minute_usd": round(sum(costs) / minutes, 3) if costs and minutes else None,
            "unpriced": sorted({m for r in rows for m in r.get("unpriced") or []}),
        }
    return summary


def render(summary: dict[str, dict[str, Any]]) -> str:
    cols = [
        ("calls", "calls"), ("users", "users"), ("minutes", "minutes"),
        ("most_minutes_by_one_user", "max min/user"), ("interruptions_per_call", "interrupts/call"),
        ("unanswered_turns", "unanswered"), ("backup_voice_calls", "backup voice"),
        ("cost_usd", "cost $"), ("cost_per_minute_usd", "$/min"), ("max_call_cost_usd", "max call $"),
    ]
    lines = ["| day | " + " | ".join(label for _, label in cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for day, s in summary.items():
        lines.append(f"| {day} | " + " | ".join("–" if s[k] is None else str(s[k]) for k, _ in cols) + " |")
    lines.append("")
    for day, s in summary.items():
        extra = [f"ended: {s['ended']}"]
        if s["vendor_failures"]:
            extra.append(f"vendor failures: {s['vendor_failures']}")
        if s["unpriced"]:
            extra.append(f"unpriced (cost too low): {', '.join(s['unpriced'])}")
        lines.append(f"{day}  " + "; ".join(extra))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    records = []
    for path in args.files:
        records += read_summaries(path.read_text())
    if not records:
        print("no voice_call_summary lines found", file=sys.stderr)
        return 1
    print(render(summarise(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
