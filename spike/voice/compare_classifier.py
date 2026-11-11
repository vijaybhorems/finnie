"""Compare classifier models: routing accuracy, scope-gate accuracy and latency.

Runs the production classify node over the labelled cases the evals already
use (tests/evals: 32 routing + 12 guardrail cases) once per configuration, and
prints accuracy, misses and p50/p95 latency side by side. A faster classifier
is only worth it if it routes and refuses as well as the current one.

    .venv/bin/python -m spike.voice.compare_classifier
Costs: one short structured call per case per configuration (well under $1).
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# (label, classifier model, thinking)
CONFIGS = [
    ("sonnet-5 / thinking on (chat today)", "claude-sonnet-5", "adaptive"),
    ("sonnet-5 / thinking off (voice today)", "claude-sonnet-5", "disabled"),
    ("haiku-5.5 / thinking off", "claude-haiku-5-5", "disabled"),
    ("haiku-5.5 / thinking on", "claude-haiku-5-5", "adaptive"),
]
RUNS = Path(__file__).parent / "runs"


def configure(model: str, thinking: str) -> None:
    from src.core.config import get_settings
    from src.core.llm import get_llm

    os.environ["FINNIE_CLASSIFIER_MODEL"] = model
    os.environ["FINNIE_LLM_THINKING"] = thinking
    get_settings.cache_clear()
    get_llm.cache_clear()


def run_case(query: str) -> tuple[dict, float]:
    from langchain_core.messages import HumanMessage

    from src.core.state import FinancialData, FinnieState, UserProfile
    from src.workflow.classify import classify_node

    state = FinnieState(messages=[HumanMessage(content=query)], user_profile=UserProfile(),
                        financial_data=FinancialData())
    start = time.perf_counter()
    result = classify_node(state)
    return result, time.perf_counter() - start


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> None:
    from src.core.state import AgentType
    from tests.evals.test_guardrail_evals import GUARDRAIL_CASES
    from tests.evals.test_router_evals import ROUTING_CASES

    RUNS.mkdir(exist_ok=True)
    summary = []
    for label, model, thinking in CONFIGS:
        configure(model, thinking)
        print(f"\n=== {label} ===")
        latencies, misses, errors = [], [], 0
        routed = 0
        for query, expected in ROUTING_CASES:
            result, seconds = run_case(query)
            latencies.append(seconds)
            reason = str(result.get("router_reasoning") or "")
            errors += reason.startswith("classifier error")
            if result["next_agent"] == expected:
                routed += 1
            else:
                misses.append(f"  route  {query!r}: expected {expected.value}, got {result['next_agent'].value}")
        gated = 0
        for query, on_topic in GUARDRAIL_CASES:
            result, seconds = run_case(query)
            if result.get("router_reasoning", "") and str(result["router_reasoning"]).startswith("classifier error"):
                errors += 1
            # Blocklist hits never reach the model, so leave them out of latency.
            if "matched blocklist" not in str(result.get("router_reasoning", "")):
                latencies.append(seconds)
            got = result["next_agent"] != AgentType.OUT_OF_SCOPE
            if got == on_topic:
                gated += 1
            else:
                misses.append(f"  scope  {query!r}: expected {'on' if on_topic else 'off'}-topic")
        row = {
            "config": label, "model": model, "thinking": thinking,
            "routing": f"{routed}/{len(ROUTING_CASES)}", "scope": f"{gated}/{len(GUARDRAIL_CASES)}",
            "errors": errors,
            "p50_s": round(statistics.median(latencies), 2), "p95_s": round(pct(latencies, 0.95), 2),
        }
        summary.append(row)
        print(f"routing {row['routing']}  scope {row['scope']}  errors {errors}  "
              f"p50 {row['p50_s']}s  p95 {row['p95_s']}s")
        print("\n".join(misses) or "  no misses")

    with (RUNS / "classifier_compare.jsonl").open("a") as fh:
        for row in summary:
            fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}) + "\n")
    print("\n| Config | Routing | Scope | Errors | p50 | p95 |\n|---|---|---|---|---|---|")
    for r in summary:
        print(f"| {r['config']} | {r['routing']} | {r['scope']} | {r['errors']} | {r['p50_s']} s | {r['p95_s']} s |")


if __name__ == "__main__":
    main()
