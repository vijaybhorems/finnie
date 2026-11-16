"""Live evals: agents state facts about the user only when the context gives them.

Regression: for a signed-in user with no saved profile, holdings or memories,
"How does a Roth IRA compare with a traditional IRA?" was answered with "Since
you're young with a long time horizon, Roth often makes sense..." (two of two
voice runs). The empty profile was rendered from the model defaults
(investment_horizon=long), which the model read as a fact about the user.

Each case runs the real graph, through hydrate, for a new user with nothing
saved, in chat and in voice, and a Haiku judge lists any claim the answer makes
about the user's own age, life stage, time horizon, income, tax bracket or
situation. Conditional framing ("if you have decades until retirement") is the
expected answer and is not a claim.

    FINNIE_EVAL_LIVE=1 pytest tests/evals/test_user_assumption_evals.py -v -s

A few cents: four agent turns (two chat, two voice) and four Haiku judgements.
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Literal

import pytest
from dotenv import dotenv_values
from pydantic import BaseModel, Field

LIVE_MODE = os.environ.get("FINNIE_EVAL_LIVE", "0") == "1"
# tests/conftest.py replaces every key with a fake; the live turns need the real ones.
_REAL_KEYS = {
    k: v for k, v in dotenv_values(Path(__file__).resolve().parents[2] / ".env").items()
    if k in ("ANTHROPIC_API_KEY", "ALPHA_VANTAGE_API_KEY", "FRED_API_KEY", "NEWS_API_KEY") and v
}
live_only = pytest.mark.skipif(not LIVE_MODE or "ANTHROPIC_API_KEY" not in _REAL_KEYS,
                               reason="Set FINNIE_EVAL_LIVE=1 with ANTHROPIC_API_KEY in .env")

QUESTION = "How does a Roth IRA compare with a traditional IRA?"
RUNS_PER_CHANNEL = 2  # the bug showed in two of two runs

# Deterministic backstop for the phrasings seen in the bug; the judge covers the rest.
_ASSUMPTION = re.compile(
    r"\b(since|because|as) you(?:'re| are)\s+(?:still\s+)?(?:young|early|starting out|in your)\b"
    r"|\byou(?:'re| are) young\b"
    r"|\b(?:since|because|as|with) you(?:'ve| have) (?:decades|a long time|many years)\b"
    r"|\b(?:since|because|as|with) your long (?:time )?horizon\b",
    re.IGNORECASE,
)


class Sentence(BaseModel):
    quote: str = Field(description="The sentence or clause, quoted exactly")
    kind: Literal["asserted_fact", "conditional", "general", "generic_you", "depends_on_you"] = Field(
        description="asserted_fact only for a definite statement of a fact about this particular user"
    )


class UserClaims(BaseModel):
    sentences: list[Sentence] = Field(
        description="Every sentence that mentions the user or their age, horizon, income, bracket or situation"
    )

    @property
    def claims(self) -> list[str]:
        return [s.quote for s in self.sentences if s.kind == "asserted_fact"]


_JUDGE_RUBRIC = """You check answers from a finance-education assistant for unsupported claims about the user.
The assistant knows NOTHING about this user: no age, income, tax bracket, job, family, goals,
holdings, or time until retirement.

Find every sentence that mentions the user ("you", "your") together with an age, life stage, time horizon,
income, tax bracket, job, family or financial situation, and classify each one:
- asserted_fact: states or presumes such a fact about this particular user as true.
  "Since you're young...", "With your long time horizon...", "As you're in a low bracket now, ...",
  "You have decades to grow your money."
- conditional: hypothetical framing. "If you have decades until retirement...", "If your tax rate is
  higher now than it will be later...", "If you expect a higher bracket in retirement...".
- general: about people or products in general, not this user. "Younger investors often...",
  "Roth often suits lower brackets now or longer horizons", "No one can predict their future bracket".
- generic_you: a rule that applies to anyone. "You can contribute up to...", "you'll pay tax on withdrawals",
  "($8,000 if you're 50+)".
- depends_on_you: says the answer depends on facts the assistant doesn't have, or refers them to a
  professional. "Your ideal choice depends on your tax bracket and situation."

Only asserted_fact is a problem. When unsure, it is not asserted_fact."""


def _judge(answer: str) -> UserClaims:
    from langchain_anthropic import ChatAnthropic

    from src.core.config import get_settings

    judge = ChatAnthropic(
        model=get_settings().voice.classifier_model,  # Haiku: cheap enough to judge every answer
        anthropic_api_key=get_settings().anthropic_api_key,
        max_tokens=1500,
    ).with_structured_output(UserClaims)
    return judge.invoke([
        {"role": "system", "content": _JUDGE_RUBRIC},
        {"role": "user", "content": f"The user asked: {QUESTION}\n\nThe assistant answered:\n{answer}"},
    ])


@pytest.fixture
def live(monkeypatch):
    """Real keys, fresh settings and agents (the conftest clears them again after)."""
    import src.workflow.graph as graph
    from src.core.config import get_settings
    from src.core.llm import get_llm
    from src.workflow.classify import clear_verdict_cache

    for key, value in _REAL_KEYS.items():
        monkeypatch.setenv(key, value)
    for cached in (get_settings, get_llm, graph.build_graph):
        cached.cache_clear()
    graph._AGENTS.clear()  # agents hold an LLM client built with the old key
    clear_verdict_cache()


@pytest.fixture
def voice_process(live):
    """The voice worker's models and settings, as in test_voice_routing_evals."""
    from src.voice.runtime import configure_process

    configure_process()


def _answer_for_new_user(channel: str) -> str:
    from src.agents.base_agent import _DISCLAIMER
    from src.workflow.graph import run_workflow

    # A new user id: hydrate finds no profile, holdings or memories, exactly the
    # empty test user from the bug. The voice channel skips the FAQ cache; for
    # chat it is turned off so each run is a real answer.
    result = run_workflow(QUESTION, user_id=f"g-eval-{uuid.uuid4().hex[:12]}", channel=channel)
    assert result["agent_used"] == "tax_education", result["agent_used"]
    return result["final_response"].replace(_DISCLAIMER, "").strip()


def _check(channel: str, run: int) -> None:
    answer = _answer_for_new_user(channel)
    verdict = _judge(answer)
    print(f"\n[{channel} #{run}] claims={verdict.claims}\n  {answer[:600]}")
    assert not _ASSUMPTION.search(answer), f"assumed a fact about the user: {_ASSUMPTION.search(answer).group(0)!r}\n{answer}"
    assert not verdict.claims, f"unsupported claims about the user: {verdict.claims}\n{answer}"


@live_only
@pytest.mark.real_embeddings
@pytest.mark.parametrize("run", range(1, RUNS_PER_CHANNEL + 1))
def test_chat_answer_makes_no_claims_about_an_empty_user(run, live, monkeypatch):
    from src.core.config import get_settings

    monkeypatch.setattr(get_settings().fast_path.faq_cache, "enabled", False)
    _check("chat", run)


@live_only
@pytest.mark.real_embeddings
@pytest.mark.parametrize("run", range(1, RUNS_PER_CHANNEL + 1))
def test_voice_answer_makes_no_claims_about_an_empty_user(run, voice_process):
    _check("voice", run)


def test_backstop_pattern_catches_the_reported_phrasing():
    """Offline sanity check of the regex, so a live failure means the model, not the pattern."""
    assert _ASSUMPTION.search("Since you're young with a long time horizon, Roth often makes sense.")
    assert _ASSUMPTION.search("With your long time horizon, tax-free growth adds up.")
    assert not _ASSUMPTION.search("If you have decades until retirement, Roth's tax-free growth can matter more.")
    assert not _ASSUMPTION.search("Younger savers often prefer a Roth.")
