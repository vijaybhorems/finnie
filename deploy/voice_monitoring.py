"""Cloud Monitoring for voice: log-based metrics from the worker's log lines, and alerts.

Design doc, Monitoring. The worker writes one JSON log line per turn
(`voice_turn`), per call (`voice_call_summary`) and per provider failure
(`voice_vendor_failed`). These metrics count them, and three alert policies
watch them:

  provider failures   Deepgram/ElevenLabs failures in 15 minutes reach a count
  slow first audio    p95 time to first audio over the last hour above a target
  daily spend         estimated voice cost over the last day above a ceiling

deploy/setup_gcp.sh's Voice alerts stage runs this, then applies the files:

    python3 deploy/voice_monitoring.py write OUT_DIR --ttfa-p95 8 --daily-spend 10 --failures 3
    gcloud logging metrics create NAME --config-from-file=OUT_DIR/metrics/NAME.json
    gcloud monitoring policies create --policy-from-file=OUT_DIR/policies/NAME.json

Standard library only, like deploy/worker_env.py. Metrics count from when they
are created: log lines from before then are not back-filled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PREFIX = "logging_googleapis_com:user_"  # a log-based metric NAME in PromQL
# The worker pool's logs. PromQL on a log-based metric must name the resource type:
# before the first data point Monitoring can't tell which one, and refuses the
# policy ("must specify a label matcher on the 'monitored_resource' label").
RESOURCE = '{monitored_resource="cloud_run_worker_pool"}'

# Seconds: fine steps around the spike's 4-8 s, since p95 is read from these.
_FIRST_AUDIO_BOUNDS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30]

METRICS: dict[str, dict[str, Any]] = {
    "voice_calls": {
        "description": "Voice calls, by why each ended (voice_call_summary).",
        "filter": 'jsonPayload.event="voice_call_summary"',
        "metricDescriptor": {
            "metricKind": "DELTA", "valueType": "INT64", "unit": "1",
            "labels": [{"key": "reason", "valueType": "STRING"}],
        },
        "labelExtractors": {"reason": "EXTRACT(jsonPayload.reason)"},
    },
    "voice_call_cost": {
        "description": "Estimated dollars per voice call, from list prices in config.yaml (voice.cost).",
        "filter": 'jsonPayload.event="voice_call_summary"',
        "metricDescriptor": {"metricKind": "DELTA", "valueType": "DISTRIBUTION", "unit": "USD"},
        "valueExtractor": "EXTRACT(jsonPayload.cost_usd)",
        "bucketOptions": {"exponentialBuckets": {"numFiniteBuckets": 20, "growthFactor": 2, "scale": 0.005}},
    },
    "voice_time_to_first_audio": {
        "description": "Seconds from the end of the caller's speech to Finnie's first audio, by agent (voice_turn).",
        "filter": 'jsonPayload.event="voice_turn" AND jsonPayload.time_to_first_audio_s>0',
        "metricDescriptor": {
            "metricKind": "DELTA", "valueType": "DISTRIBUTION", "unit": "s",
            "labels": [{"key": "agent", "valueType": "STRING"}],
        },
        "valueExtractor": "EXTRACT(jsonPayload.time_to_first_audio_s)",
        "labelExtractors": {"agent": "EXTRACT(jsonPayload.agent)"},
        "bucketOptions": {"explicitBuckets": {"bounds": _FIRST_AUDIO_BOUNDS}},
    },
    "voice_vendor_failures": {
        "description": "Speech provider failures, at most one per vendor per call (voice_vendor_failed).",
        "filter": 'jsonPayload.event="voice_vendor_failed"',
        "metricDescriptor": {
            "metricKind": "DELTA", "valueType": "INT64", "unit": "1",
            "labels": [{"key": "vendor", "valueType": "STRING"}],
        },
        "labelExtractors": {"vendor": "EXTRACT(jsonPayload.vendor)"},
    },
}


def _policy(name: str, title: str, query: str, docs: str, channel: str) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "displayName": f"Finnie voice: {title}",
        "userLabels": {"app": "finnie", "alert": name.replace("_", "-")},
        "combiner": "OR",
        "conditions": [{
            "displayName": title,
            "conditionPrometheusQueryLanguage": {"query": query, "duration": "0s", "evaluationInterval": "60s"},
        }],
        "documentation": {"content": docs, "mimeType": "text/markdown"},
        "alertStrategy": {"autoClose": "86400s"},
        "severity": "WARNING",
    }
    if channel:
        policy["notificationChannels"] = [channel]
    return policy


def policies(ttfa_p95: float, daily_spend: float, failures: int, channel: str = "") -> dict[str, dict[str, Any]]:
    """The three alert policies, by file name. `channel` is a notification channel's name."""
    if ttfa_p95 <= 0 or daily_spend <= 0 or failures < 1:
        raise SystemExit("thresholds must be positive")
    report = "Per-day figures: `python scripts/voice_report.py` on an export of the voice_call_summary lines."
    return {
        "voice_provider_failures": _policy(
            "voice_provider_failures", f"{failures}+ provider failures in 15 min",
            f"sum(increase({PREFIX}voice_vendor_failures{RESOURCE}[15m])) >= {failures}",
            "Deepgram or ElevenLabs failed on several calls. Callers may hear \"I'm having trouble "
            "hearing you\" or the backup voice. Check the vendors' status pages and the worker's "
            "`voice_vendor_failed` lines (`vendor`, `what`).",
            channel,
        ),
        "voice_slow_first_audio": _policy(
            "voice_slow_first_audio", f"p95 first audio above {ttfa_p95:g} s (1 h)",
            f"histogram_quantile(0.95, sum by (le) (increase({PREFIX}voice_time_to_first_audio_bucket{RESOURCE}[1h]))) "
            f"> {ttfa_p95:g}",
            "Finnie is slow to start speaking. By step and agent: export the `voice_turn` lines and run "
            "`python scripts/voice_latency.py` on them.",
            channel,
        ),
        "voice_daily_spend": _policy(
            "voice_daily_spend", f"voice spend above ${daily_spend:g} in a day",
            f"sum(increase({PREFIX}voice_call_cost_sum{RESOURCE}[1d])) > {daily_spend:g}",
            "Estimated voice cost (list prices in config.yaml, voice.cost) passed the daily ceiling. "
            + report + " Lower voice.daily_minutes_per_user or narrow the allowlist to cut it.",
            channel,
        ),
    }


def write(out: Path, ttfa_p95: float, daily_spend: float, failures: int, channel: str = "") -> list[Path]:
    written = []
    for kind, files in (("metrics", METRICS), ("policies", policies(ttfa_p95, daily_spend, failures, channel))):
        (out / kind).mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            path = out / kind / f"{name}.json"
            path.write_text(json.dumps(body, indent=2) + "\n")
            written.append(path)
    return written


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="write metrics/*.json and policies/*.json into OUT_DIR")
    w.add_argument("out", type=Path)
    w.add_argument("--ttfa-p95", type=float, default=8.0, help="seconds")
    w.add_argument("--daily-spend", type=float, default=10.0, help="US dollars")
    w.add_argument("--failures", type=int, default=3, help="in 15 minutes")
    w.add_argument("--channel", default="", help="notification channel name (projects/…/notificationChannels/…)")
    args = parser.parse_args(argv[1:])
    for path in write(args.out, args.ttfa_p95, args.daily_spend, args.failures, args.channel):
        print(path)


if __name__ == "__main__":
    main(sys.argv)
