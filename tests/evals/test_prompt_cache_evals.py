"""Live prompt-cache evals — the standing check that caching actually works.

Caching fails silently: requests still succeed, they just cost more. The API's
usage fields are the only ground truth, so these tests send real requests and
assert on them. They use max_tokens=16 because they measure the prompt, not the
answer — a full run costs a few cents.

    FINNIE_EVAL_LIVE=1 pytest tests/evals/test_prompt_cache_evals.py -v -s

Re-run after any change to prompt assembly (src/agents/base_agent.py,
src/agents/prompts/, src/rag/digest.py) — a new per-request value in a stable
block is the classic regression, and nothing else will surface it.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage

from src.core.llm import token_usage
from src.core.state import FinancialData, FinnieState, UserProfile

LIVE_MODE = os.environ.get("FINNIE_EVAL_LIVE", "0") == "1"
_ROOT = Path(__file__).resolve().parents[2]

# Minimum cacheable prefix per model; shorter prefixes silently never cache.
_MIN_CACHEABLE_TOKENS = {
    "claude-sonnet-5": 1024,
    "claude-sonnet-5-5": 512,
    "claude-opus-5": 512,
    "claude-opus-5-5": 512,
}


def _real_api_key() -> str:
    # tests/conftest.py replaces ANTHROPIC_API_KEY with a fake for every test,
    # so read the real key from .env directly.
    return dotenv_values(_ROOT / ".env").get("ANTHROPIC_API_KEY") or ""


pytestmark = pytest.mark.skipif(
    not LIVE_MODE or not _real_api_key(),
    reason="Set FINNIE_EVAL_LIVE=1 and ANTHROPIC_API_KEY in .env for live cache evals",
)


def _model() -> str:
    from src.core.config import get_settings

    return get_settings().llm.model


def _llm():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=_model(), max_tokens=16, anthropic_api_key=_real_api_key())


def _state(query: str) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(),
        financial_data=FinancialData(),
    )


def _agent(kind: str):
    with patch("src.agents.finance_qa_agent.get_retriever"), \
         patch("src.agents.finance_qa_agent.FredClient"), \
         patch("src.agents.tax_education_agent.get_retriever"), \
         patch("src.agents.base_agent.get_llm"):
        if kind == "finance_qa":
            from src.agents.finance_qa_agent import FinanceQAAgent
            return FinanceQAAgent()
        from src.agents.tax_education_agent import TaxEducationAgent
        return TaxEducationAgent()


def _system_tokens(text: str) -> int:
    """Tokens a system block adds, via the count_tokens endpoint (free)."""
    import anthropic

    client = anthropic.Anthropic(api_key=_real_api_key())
    probe = [{"role": "user", "content": "hi"}]
    with_system = client.messages.count_tokens(
        model=_model(), system=[{"type": "text", "text": text}], messages=probe
    )
    baseline = client.messages.count_tokens(model=_model(), messages=probe)
    return with_system.input_tokens - baseline.input_tokens


def test_shared_block_clears_the_minimum_on_its_own():
    """Below the minimum, the shared breakpoint never caches and agents can't share an entry."""
    from src.agents.prompts import shared_system_prompt

    tokens = _system_tokens(shared_system_prompt())
    minimum = _MIN_CACHEABLE_TOKENS.get(_model(), 1024)
    print(f"\nshared block: {tokens} tokens (minimum {minimum} on {_model()})")
    assert tokens >= minimum


def test_second_request_reads_the_cache():
    """Same agent, different question and data: the stable prefix must be read back."""
    agent = _agent("finance_qa")
    llm = _llm()

    first = token_usage(llm.invoke(agent._build_messages(_state("What is a P/E ratio?"), "ctx: A")))
    second = token_usage(llm.invoke(agent._build_messages(_state("How do bonds work?"), "ctx: B")))
    print(f"\nfirst:  {first}\nsecond: {second}")

    assert second["cache_read_input_tokens"] > 0, (
        "Second request read nothing from cache — a stable block is changing between "
        "requests, or the prefix is below the model's minimum cacheable length."
    )
    assert second["cache_creation_input_tokens"] == 0


def test_shared_block_is_reused_across_agents():
    """A different agent's first call should read the shared block another agent wrote."""
    from src.agents.prompts import shared_system_prompt

    llm = _llm()
    llm.invoke(_agent("finance_qa")._build_messages(_state("What is a P/E ratio?"), "ctx"))
    tax = token_usage(llm.invoke(_agent("tax")._build_messages(_state("What is a Roth IRA?"), "ctx")))
    shared = _system_tokens(shared_system_prompt())
    print(f"\ntax after finance_qa: {tax} (shared block ~{shared} tokens)")

    assert tax["cache_read_input_tokens"] >= shared * 0.9
