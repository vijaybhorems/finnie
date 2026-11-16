"""Tune turn detection with simulated calls (design doc: turn detection, Evaluation layer 5).

For each combination of the `--grid` voice settings, starts a worker with it and
plays the turn-taking scenarios (tests/evals/sim_calls.py, `turn_taking=True`:
pauses mid-question, "uh-huh" while Finnie talks, real cut-ins) `--repeat` times.
Reports, per setting:
  split     calls where the graph ran a different number of times than expected:
            a question answered in pieces, or a backchannel answered as a turn
  missed    calls where a real cut-in wasn't counted, or a backchannel was
  failed    calls with any failed check
  first audio p50/p95 from the caller's clock (end of its speech to Finnie's first sound)
A longer endpointing min delay splits fewer questions and answers later; more
interruption words ignore "uh-huh" and a lone "Stop." alike.

Run with the voice venv, the local LiveKit server up and no other worker:
    .venv-voice/bin/python scripts/voice_turn_sweep.py --grid endpointing_min_delay=0.4,0.5,0.8 --repeat 2
    .venv-voice/bin/python scripts/voice_turn_sweep.py --grid interruption_min_words=0,2
Costs about 5-10 cents a call: settings x scenarios x repeat (the first line: 30 calls).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_grid(items: list[str]) -> list[dict[str, Any]]:
    """["a=1,2", "b=0.5"] -> [{"a": 1, "b": 0.5}, {"a": 2, "b": 0.5}]; names must be voice settings."""
    from src.core.config import VoiceConfig

    axes: dict[str, list[Any]] = {}
    for item in items:
        name, _, values = item.partition("=")
        if name not in VoiceConfig.model_fields:
            raise SystemExit(f"--grid: {name!r} is not a voice setting")
        axes[name] = [json.loads(v) for v in values.split(",") if v]
    return [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarise(setting: dict[str, float], calls: list[dict[str, Any]]) -> dict[str, Any]:
    def failed(call: dict[str, Any], prefix: str) -> bool:
        return any(why for name, why in call["checks"].items() if name.startswith(prefix))

    audio = [s for c in calls for s in c["first_audio_s"] if s is not None]
    return {
        **setting,
        "calls": len(calls),
        "split": sum(failed(c, "graph_runs") for c in calls),
        "missed": sum(failed(c, "interruptions") for c in calls),
        "failed": sum(any(c["checks"].values()) for c in calls),
        "first_audio_p50": statistics.median(audio) if audio else None,
        "first_audio_p95": _pct(audio, 0.95),
    }


TALLY = ("calls", "split", "missed", "failed", "first_audio_p50", "first_audio_p95")


def render(rows: list[dict[str, Any]]) -> str:
    names = [k for k in rows[0] if k not in TALLY] if rows else []
    widths = [max(len(n), 6) for n in names]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else ("-" if v is None else str(v))  # noqa: E731
    head = " ".join(n.rjust(w) for n, w in zip(names, widths)) + "  calls split missed failed    p50    p95"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(" ".join(fmt(r[n]).rjust(w) for n, w in zip(names, widths))
                     + f"  {r['calls']:5d} {r['split']:5d} {r['missed']:6d} {r['failed']:6d} "
                     + f"{fmt(r['first_audio_p50']):>6} {fmt(r['first_audio_p95']):>6}")
    return "\n".join(lines)


async def _sweep_one(setting: dict[str, float], scenarios: list, repeat: int) -> list[dict[str, Any]]:
    from src.core.config import get_settings
    from tests.evals.sim_caller import Worker, run_scenario
    from tests.evals.sim_calls import EVAL_WORKER_SETTINGS, run_checks

    worker = Worker({**EVAL_WORKER_SETTINGS, **setting})
    calls = []
    try:
        await asyncio.to_thread(worker.wait_ready)
        for n, scenario in itertools.product(range(repeat), scenarios):
            get_settings.cache_clear()
            observed = await run_scenario(scenario, worker, tag=f"r{n}")
            checks = run_checks(scenario, observed)
            calls.append({"scenario": scenario.name, "checks": checks, "first_audio_s": observed.first_audio_s,
                          "questions": [t.get("question") for t in observed.turns]})
            bad = {k: v for k, v in checks.items() if v}
            print(f"  {scenario.name:<24} {'ok' if not bad else bad}", flush=True)
    finally:
        worker.stop()
    return calls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", action="append", default=[], metavar="SETTING=V1,V2",
                        help="a voice setting and the values to try (repeatable; combinations are crossed)")
    parser.add_argument("--repeat", type=int, default=1, help="plays of each scenario per setting")
    parser.add_argument("--scenarios", default="", help="names (default: every turn-taking scenario)")
    parser.add_argument("--out", type=Path, default=ROOT / "tests/evals/.sim_runs/turn_sweep.json")
    args = parser.parse_args(argv)
    # The caller mints through the app's minter in this process (spike/voice/local_token.py).
    os.environ.setdefault("FINNIE_VOICE_ENABLED", "1")
    os.environ.setdefault("FINNIE_VOICE_ALLOWED_EMAILS", "tester@example.com")
    os.environ["DATABASE_URL"] = ""

    from tests.evals.sim_calls import SCENARIOS, by_name

    scenarios = by_name(args.scenarios.split(",")) if args.scenarios else [s for s in SCENARIOS if s.turn_taking]
    settings = parse_grid(args.grid or ["endpointing_min_delay=0.5"])
    print(f"{len(settings)} settings x {len(scenarios)} scenarios x {args.repeat} = "
          f"{len(settings) * len(scenarios) * args.repeat} calls", flush=True)

    rows, raw = [], []
    for setting in settings:
        print("  ".join(f"{k}={v}" for k, v in setting.items()), flush=True)
        calls = asyncio.run(_sweep_one(setting, scenarios, args.repeat))
        rows.append(summarise(setting, calls))
        raw.append({"setting": setting, "calls": calls})
    print("\n" + render(rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "runs": raw}, indent=2, default=str))
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
