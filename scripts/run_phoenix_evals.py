"""Phoenix-based evals for prompt quality and agent routing.

Two evaluations:
  1. Routing accuracy  — runs the real router over labelled cases and reports
     accuracy + a per-miss breakdown (debug agent routing).
  2. Prompt/answer quality — uses phoenix.evals.llm_classify with an LLM judge
     to label agent answers (evaluate prompts).

These are LLM-as-judge / live evals: they call the real Anthropic API and, for
the quality eval, require ``arize-phoenix`` to be installed. Run on demand, not
in CI.

Usage:
    python scripts/run_phoenix_evals.py --routing
    python scripts/run_phoenix_evals.py --quality
    python scripts/run_phoenix_evals.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.core.tracing import setup_tracing
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def run_routing_eval() -> float:
    """Run the router over labelled cases and report accuracy. Returns accuracy [0,1]."""
    from src.core.state import FinancialData, FinnieState, UserProfile
    from src.workflow.router import router_node
    from tests.evals.test_router_evals import ROUTING_CASES
    from langchain_core.messages import HumanMessage

    total = len(ROUTING_CASES)
    correct = 0
    misses: list[str] = []

    for query, expected in ROUTING_CASES:
        state = FinnieState(
            messages=[HumanMessage(content=query)],
            user_profile=UserProfile(),
            financial_data=FinancialData(),
        )
        result = router_node(state)
        got = result["next_agent"]
        if got == expected:
            correct += 1
        else:
            misses.append(
                f"  MISS: {query!r}\n    expected={expected.value} got={got.value} "
                f"reason={result.get('router_reasoning', '')}"
            )

    accuracy = correct / total if total else 0.0
    print(f"\n=== Routing accuracy: {correct}/{total} = {accuracy:.1%} ===")
    if misses:
        print("\n".join(misses))
    return accuracy


# Small labelled set of finance questions for the answer-quality judge.
_QUALITY_QUERIES = [
    "What is a P/E ratio and why does it matter?",
    "How does compound interest grow my savings?",
    "What is the difference between an ETF and a mutual fund?",
    "How does a Roth IRA work?",
    "What happens to bond prices when interest rates rise?",
]


def run_quality_eval() -> None:
    """Judge answer quality of the Finance Q&A agent via phoenix.evals.llm_classify."""
    try:
        import pandas as pd
        from phoenix.evals import AnthropicModel, llm_classify
    except ImportError as exc:
        logger.error("phoenix_evals_unavailable", error=str(exc))
        print(
            "phoenix.evals not installed. Install with: "
            "pip install arize-phoenix pandas"
        )
        return

    from src.agents.finance_qa_agent import FinanceQAAgent
    from src.core.config import get_settings
    from src.core.state import FinancialData, FinnieState, UserProfile
    from langchain_core.messages import HumanMessage

    settings = get_settings()
    agent = FinanceQAAgent()

    rows = []
    for query in _QUALITY_QUERIES:
        state = FinnieState(
            messages=[HumanMessage(content=query)],
            user_profile=UserProfile(),
            financial_data=FinancialData(),
        )
        result = agent.run(state)
        rows.append({"input": query, "output": result.get("final_response", "")})

    df = pd.DataFrame(rows)

    # Custom rubric: is the answer a correct, on-topic, non-advisory finance explanation?
    template = (
        "You are grading answers from a finance education assistant.\n"
        "[Question]: {input}\n"
        "[Answer]: {output}\n\n"
        "Is the answer factually correct, on-topic for personal finance, and "
        "educational (not giving specific buy/sell advice)?\n"
        'Respond with a single word, "correct" or "incorrect".'
    )

    model = AnthropicModel(
        model=settings.llm.model,
        api_key=settings.anthropic_api_key,
    )

    result_df = llm_classify(
        dataframe=df,
        template=template,
        model=model,
        rails=["correct", "incorrect"],
        provide_explanation=True,
    )

    merged = df.join(result_df[["label", "explanation"]])
    correct = (merged["label"] == "correct").sum()
    print(f"\n=== Answer quality: {correct}/{len(merged)} correct ===")
    for _, row in merged.iterrows():
        print(f"  [{row['label']}] {row['input']}\n    {row.get('explanation', '')}")


def main() -> None:
    """Parse CLI arguments and run the selected Phoenix evals."""
    setup_logging()
    setup_tracing()  # export eval-run spans if a Phoenix endpoint is configured

    parser = argparse.ArgumentParser(description="Run Phoenix-based Finnie evals.")
    parser.add_argument("--routing", action="store_true", help="Run routing accuracy eval")
    parser.add_argument("--quality", action="store_true", help="Run answer-quality eval")
    parser.add_argument("--all", action="store_true", help="Run all evals")
    args = parser.parse_args()

    if not (args.routing or args.quality or args.all):
        parser.error("Specify at least one of --routing, --quality, or --all")

    if args.routing or args.all:
        run_routing_eval()
    if args.quality or args.all:
        run_quality_eval()


if __name__ == "__main__":
    main()
