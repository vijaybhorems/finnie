"""scripts/voice_turn_sweep.py's tally and table, on made-up calls (no LiveKit)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "voice_turn_sweep", Path(__file__).resolve().parents[1] / "scripts" / "voice_turn_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

SETTING = {"endpointing_min_delay": 0.5, "endpointing_max_delay": 3.0}


def call(checks, audio=(2.0,)):
    return {"checks": checks, "first_audio_s": list(audio)}


def test_counts_splits_missed_cut_ins_and_failures():
    calls = [
        call({"graph_runs(1)": None, "heard(roth)": None}, (1.0,)),
        call({"graph_runs(1)": "2 graph runs, expected 1", "heard(roth)": None}, (3.0,)),
        call({"graph_runs(2)": None, "interruptions(1..)": "0 interruptions"}, (2.0, None)),
        call({"graph_runs(1)": None, "heard(roth)": "missing"}, (4.0,)),
    ]
    row = sweep.summarise(SETTING, calls)
    assert (row["calls"], row["split"], row["missed"], row["failed"]) == (4, 1, 1, 3)
    assert row["first_audio_p50"] == 2.5 and row["first_audio_p95"] == 4.0


def test_no_audio_measured():
    row = sweep.summarise(SETTING, [call({"graph_runs(1)": None}, (None,))])
    assert row["first_audio_p50"] is None and row["failed"] == 0
    assert "-" in sweep.render([row])


def test_table_has_a_row_per_setting():
    rows = [sweep.summarise({**SETTING, "endpointing_min_delay": lo}, [call({})]) for lo in (0.4, 0.8)]
    table = sweep.render(rows).splitlines()
    assert table[0].split()[:2] == ["endpointing_min_delay", "endpointing_max_delay"]
    assert len(table) == 4 and table[2].split()[0] == "0.40" and table[3].split()[0] == "0.80"


def test_grid_crosses_settings_and_keeps_types():
    grid = sweep.parse_grid(["interruption_min_words=0,2", "endpointing_min_delay=0.5"])
    assert grid == [{"interruption_min_words": 0, "endpointing_min_delay": 0.5},
                    {"interruption_min_words": 2, "endpointing_min_delay": 0.5}]


def test_grid_refuses_unknown_settings():
    import pytest

    with pytest.raises(SystemExit, match="not a voice setting"):
        sweep.parse_grid(["endpointing_min_dealy=0.5"])
