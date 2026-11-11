"""Measure Finnie's share of voice latency, with no voice vendors involved.

Time to first audio = turn decision + Finnie (classifier, data, first sentence)
+ text-to-speech first byte. This script times the middle part, which is the
largest and the only one we can change in our own code, using the real
Anthropic key from .env. Each question runs `--repeats` times, so FAQ-cacheable
questions also produce cache-hit samples.

Run with the main venv (no LiveKit needed):
    .venv/bin/python -m spike.voice.measure_turns --repeats 2
Costs a few Anthropic calls per question (classifier timed separately, then
the full turn).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Representative spoken questions by turn type (design doc, Open question 6).
QUESTIONS = [
    ("qa_tax_goals", "What is a P/E ratio?"),
    ("qa_tax_goals", "How does a Roth IRA work?"),
    ("qa_tax_goals", "How much do I need to save to retire at 55?"),
    ("live_data", "How is Apple stock doing today?"),
    ("live_data", "What's the latest news about the Fed?"),
    ("live_data", "How is the energy sector doing today?"),
]

RUNS = Path(__file__).parent / "runs"


def time_classifier(question: str) -> float:
    from langchain_core.messages import HumanMessage

    from src.core.state import FinancialData, FinnieState, UserProfile
    from src.workflow.classify import classify_node

    state = FinnieState(
        messages=[HumanMessage(content=question)],
        user_profile=UserProfile(),
        financial_data=FinancialData(),
    )
    start = time.perf_counter()
    classify_node(state)
    return time.perf_counter() - start


def time_turn(question: str, channel: str) -> dict:
    from spike.voice.speech import SpeechBuffer, strip_written_disclaimer
    from src.workflow.graph import stream_workflow

    sink: dict = {}
    buffer = SpeechBuffer()
    start = time.perf_counter()
    first_token = first_sentence = None
    for token in stream_workflow(question, sink=sink, channel=channel):
        now = time.perf_counter()
        first_token = first_token or now
        if first_sentence is None and buffer.push(strip_written_disclaimer(token)).strip():
            first_sentence = now
    end = time.perf_counter()
    return {
        "agent": sink.get("agent_used"),
        "cache_hit": bool(sink.get("cache_hit")),
        "first_token_s": round((first_token or end) - start, 3),
        # What text-to-speech waits for: the first cleaned, speakable piece.
        "first_sentence_s": round((first_sentence or end) - start, 3),
        "total_s": round(end - start, 3),
        "answer_chars": len(sink.get("final_response", "")),
    }


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--channel", choices=["chat", "voice"], default="chat",
                        help="voice: short spoken answers, no FAQ cache (what the worker sends)")
    args = parser.parse_args()

    from src.web_app import warmup

    print("Warming up (embedding model, graph)...")
    warmup.wait()

    RUNS.mkdir(exist_ok=True)
    out = RUNS / "finnie_turns.jsonl"
    rows = []
    for turn_type, question in QUESTIONS:
        for attempt in range(args.repeats):
            classifier_s = round(time_classifier(question), 3)
            row = {"turn_type": turn_type, "question": question, "attempt": attempt + 1,
                   "channel": args.channel, "classifier_s": classifier_s,
                   **time_turn(question, args.channel)}
            if row["cache_hit"]:
                row["turn_type"] = "faq_hit"
            rows.append(row)
            print(f"  {row['turn_type']:13} {row['agent'] or '-':17} classifier {classifier_s:5.2f}s"
                  f"  first sentence {row['first_sentence_s']:5.2f}s  total {row['total_s']:5.2f}s"
                  f"  {row['answer_chars']:5} chars"
                  f"  {question}")
    with out.open("a") as fh:
        for row in rows:
            fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}) + "\n")

    print("\nFinnie's share of time to first audio (first speakable sentence), by turn type:")
    for turn_type in ("faq_hit", "qa_tax_goals", "live_data"):
        sample = [r for r in rows if r["turn_type"] == turn_type]
        if not sample:
            continue
        firsts = [r["first_sentence_s"] for r in sample]
        classes = [r["classifier_s"] for r in sample if not r["cache_hit"]]
        cls = f"classifier p50 {statistics.median(classes):.2f}s" if classes else "no classifier call"
        print(f"  {turn_type:13} n={len(sample):2}  p50 {statistics.median(firsts):.2f}s"
              f"  p95 {pct(firsts, 0.95):.2f}s  ({cls})")
    print(f"\nRows appended to {out}")


if __name__ == "__main__":
    main()
