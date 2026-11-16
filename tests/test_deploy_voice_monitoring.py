"""deploy/voice_monitoring.py: log-based metrics and alert policies match what the worker logs."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "voice_monitoring.py"
_spec = importlib.util.spec_from_file_location("voice_monitoring", SCRIPT)
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)

WORKER = (ROOT / "src" / "voice" / "worker.py").read_text()


def test_every_metric_reads_an_event_and_fields_the_worker_logs():
    from src.core.config import VoiceCostConfig
    from src.voice.usage import call_summary

    summary_fields = set(call_summary(
        user_id="g-1", reason="r", seconds=1, turns=[], interruptions=0, stt_seconds=None,
        tts_chars={}, tokens={}, vendor_failures=[], backup_voice_used=False, rates=VoiceCostConfig(),
    ))
    fields_by_event = {
        "voice_call_summary": summary_fields,
        "voice_turn": {"time_to_first_audio_s", "agent"},
        "voice_vendor_failed": {"vendor", "what"},
    }
    for name, metric in vm.METRICS.items():
        event = re.search(r'jsonPayload\.event="(\w+)"', metric["filter"]).group(1)
        assert f'"{event}"' in WORKER, f"{name}: the worker never logs {event}"
        used = re.findall(r"jsonPayload\.(\w+)", json.dumps(metric))
        assert set(used) - {"event"} <= fields_by_event[event], name
    # voice_turn's own fields are built in the worker: check they exist there too.
    for field in ("time_to_first_audio_s", '"agent"'):
        assert field in WORKER


def test_every_alert_queries_a_defined_metric():
    distributions = {n for n, m in vm.METRICS.items() if m["metricDescriptor"]["valueType"] == "DISTRIBUTION"}
    for name, policy in vm.policies(8, 10, 3).items():
        query = policy["conditions"][0]["conditionPrometheusQueryLanguage"]["query"]
        for metric in re.findall(re.escape(vm.PREFIX) + r"(\w+)", query):
            base = re.sub(r"_(bucket|sum|count)$", "", metric)
            assert base in vm.METRICS, (name, metric)
            assert (base != metric) == (base in distributions), f"{name}: suffix must match the value type"


def test_thresholds_and_channel_reach_the_policies():
    p = vm.policies(6.5, 25, 5, channel="projects/p/notificationChannels/9")
    queries = {n: x["conditions"][0]["conditionPrometheusQueryLanguage"]["query"] for n, x in p.items()}
    assert queries["voice_slow_first_audio"].endswith("> 6.5")
    assert queries["voice_daily_spend"].endswith("> 25")
    assert queries["voice_provider_failures"].endswith(">= 5")
    assert all(x["notificationChannels"] == ["projects/p/notificationChannels/9"] for x in p.values())
    assert "notificationChannels" not in vm.policies(8, 10, 3)["voice_daily_spend"]


def test_nonsense_thresholds_are_refused():
    with pytest.raises(SystemExit):
        vm.policies(0, 10, 3)
    with pytest.raises(SystemExit):
        vm.policies(8, 10, 0)


def test_writes_one_file_per_metric_and_policy(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "write", str(tmp_path), "--daily-spend", "20"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert sorted(Path(p).name for p in out) == sorted(
        [f"{n}.json" for n in vm.METRICS] + [f"{n}.json" for n in vm.policies(8, 10, 3)]
    )
    spend = json.loads((tmp_path / "policies" / "voice_daily_spend.json").read_text())
    assert spend["conditions"][0]["conditionPrometheusQueryLanguage"]["query"].endswith("> 20")
