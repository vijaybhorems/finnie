"""Voice-mode text evals (design doc, Evaluation layer 2). No audio.

Two questions, both against the real Claude API (FINNIE_EVAL_LIVE=1, a few cents):

1. Routing: the router evals' cases rewritten the way speech-to-text hands them
   over (lowercase, no punctuation, filler words, numbers spelled out, "apple"
   not AAPL), routed by the same classifier the voice worker uses
   (`configure_process()`: voice.classifier_model, thinking off). Company names
   must still resolve to tickers, since voice users never say symbols.
2. Answers: one voice turn per agent through `run_workflow(channel="voice")`.
   Length, markdown and dollar amounts (discreet mode, the default) are checked
   in code; a Haiku judge scores only how natural the answer sounds aloud.

Run with:
    FINNIE_EVAL_LIVE=1 pytest tests/evals/test_voice_routing_evals.py -v -s

The judge makes six direct calls. Once the case list grows into the hundreds,
send it through the Message Batches API (about half price), as the doc suggests.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values
from pydantic import BaseModel, Field

from src.core.state import AgentType
from src.voice.speech import AMOUNT_ON_SCREEN
from tests.evals.test_router_evals import ROUTING_CASES, _make_state, route_like_production

LIVE_MODE = os.environ.get("FINNIE_EVAL_LIVE", "0") == "1"
# tests/conftest.py replaces every key with a fake; the live turns need the real ones.
_REAL_KEYS = {
    k: v for k, v in dotenv_values(Path(__file__).resolve().parents[2] / ".env").items()
    if k in ("ANTHROPIC_API_KEY", "ALPHA_VANTAGE_API_KEY", "FRED_API_KEY", "NEWS_API_KEY") and v
}
pytestmark = [
    pytest.mark.skipif(not LIVE_MODE or "ANTHROPIC_API_KEY" not in _REAL_KEYS,
                       reason="Set FINNIE_EVAL_LIVE=1 with ANTHROPIC_API_KEY in .env"),
    pytest.mark.real_embeddings,
]

# ROUTING_CASES, as a transcript of someone saying them. Same order, same agents.
VOICE_ROUTING_CASES = [
    ("um what is a p e ratio", AgentType.FINANCE_QA),
    ("can you explain compound interest to me", AgentType.FINANCE_QA),
    ("whats the difference between stocks and bonds", AgentType.FINANCE_QA),
    ("so how does inflation affect my money", AgentType.FINANCE_QA),
    ("what is an expense ratio", AgentType.FINANCE_QA),
    ("define dollar cost averaging", AgentType.FINANCE_QA),

    ("analyze my portfolio i have ten shares of apple at a hundred fifty and five shares of microsoft "
     "at three hundred", AgentType.PORTFOLIO),
    ("how diversified is my portfolio", AgentType.PORTFOLIO),
    ("whats the beta of my holdings", AgentType.PORTFOLIO),
    ("uh check how my portfolio is doing", AgentType.PORTFOLIO),

    ("what is apple stock doing today", AgentType.MARKET_ANALYSIS),
    ("show me the r s i for tesla", AgentType.MARKET_ANALYSIS),
    ("whats the current price of nvidia", AgentType.MARKET_ANALYSIS),
    ("how are tech stocks performing", AgentType.MARKET_ANALYSIS),
    ("is there a golden cross on the s and p five hundred", AgentType.MARKET_ANALYSIS),
    ("how is the energy sector doing today", AgentType.MARKET_ANALYSIS),

    ("how much do i need to retire at fifty five", AgentType.GOAL_PLANNING),
    ("can i save a hundred thousand dollars in five years", AgentType.GOAL_PLANNING),
    ("i want to buy a house in three years how much should i save", AgentType.GOAL_PLANNING),
    ("what is the fire movement and how do i calculate my number", AgentType.GOAL_PLANNING),
    ("help me plan my retirement savings", AgentType.GOAL_PLANNING),

    ("what happened in the market today", AgentType.NEWS_SYNTHESIZER),
    ("any news about the fed meeting", AgentType.NEWS_SYNTHESIZER),
    ("what are the latest earnings reports", AgentType.NEWS_SYNTHESIZER),
    ("tell me about recent s e c filings", AgentType.NEWS_SYNTHESIZER),
    ("did anything big move markets this week", AgentType.NEWS_SYNTHESIZER),

    ("how does a roth i r a work", AgentType.TAX_EDUCATION),
    ("what are capital gains taxes", AgentType.TAX_EDUCATION),
    ("should i do tax loss harvesting", AgentType.TAX_EDUCATION),
    ("what is the twenty twenty four four oh one k contribution limit", AgentType.TAX_EDUCATION),
    ("how does an h s a work for investing", AgentType.TAX_EDUCATION),
    ("whats the wash sale rule", AgentType.TAX_EDUCATION),
]

# Spoken company names the classifier must turn into symbols.
SPOKEN_TICKERS = [
    ("what is apple stock doing today", "AAPL"),
    ("show me the r s i for tesla", "TSLA"),
    ("whats the current price of nvidia", "NVDA"),
    ("how is microsoft doing this week", "MSFT"),
    ("any news on amazon", "AMZN"),
]

# One voice turn per agent. The portfolio and goals turns involve the user's own
# money, which discreet mode keeps off the audio.
PORTFOLIO = [
    {"ticker": "AAPL", "shares": 50, "avg_cost": 150},
    {"ticker": "MSFT", "shares": 20, "avg_cost": 300},
    {"ticker": "VTI", "shares": 40, "avg_cost": 200},
]
ANSWER_CASES = [
    ("what is a p e ratio", AgentType.FINANCE_QA, None),
    ("how is my portfolio doing", AgentType.PORTFOLIO, {"portfolio": PORTFOLIO}),
    ("how is apple stock doing today", AgentType.MARKET_ANALYSIS, None),
    ("if i save two thousand a month at seven percent how much will i have in twenty years",
     AgentType.GOAL_PLANNING, None),
    ("how does a roth i r a work", AgentType.TAX_EDUCATION, None),
    ("what happened in the market today", AgentType.NEWS_SYNTHESIZER, None),
]
MAX_WORDS = 90  # the voice prompt asks for about 60
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*+]\s|^\s*\d+[.)]\s|\]\(|```|\|)", re.MULTILINE)
_AMOUNT = re.compile(r"\$\s?\d|\b\d[\d,.]*\s*(?:k|thousand|million|billion)?\s*dollars\b", re.IGNORECASE)


@pytest.fixture(autouse=True)
def voice_process(monkeypatch):
    """Route and answer with the voice worker's models (the conftest clears settings after)."""
    import src.workflow.graph as graph
    from src.core.config import get_settings
    from src.core.llm import get_llm
    from src.voice.runtime import configure_process
    from src.workflow.classify import clear_verdict_cache

    for key, value in _REAL_KEYS.items():
        monkeypatch.setenv(key, value)
    for cached in (get_settings, get_llm, graph.build_graph):
        cached.cache_clear()
    graph._AGENTS.clear()  # agents hold an LLM client built with the old key
    clear_verdict_cache()
    configure_process()
    return get_settings()


def test_cases_mirror_the_router_evals():
    assert [agent for _, agent in VOICE_ROUTING_CASES] == [agent for _, agent in ROUTING_CASES]


@pytest.mark.parametrize("query,expected", VOICE_ROUTING_CASES, ids=[q[:40] for q, _ in VOICE_ROUTING_CASES])
def test_spoken_routing(query, expected, voice_process):
    result = route_like_production(_make_state(query))
    assert result["next_agent"] == expected, (
        f"'{query}' routed to {result['next_agent'].value}, expected {expected.value} "
        f"(classifier {voice_process.llm.classifier_model}): {result.get('router_reasoning', '')}"
    )


@pytest.mark.parametrize("query,ticker", SPOKEN_TICKERS)
def test_spoken_company_names_resolve_to_tickers(query, ticker):
    result = route_like_production(_make_state(query))
    assert ticker in (result.get("query_tickers") or []), result


class Naturalness(BaseModel):
    score: int = Field(ge=1, le=5, description="5: sounds like a knowledgeable person talking; 1: reads like a document")
    reason: str


_JUDGE_RUBRIC = """You judge answers a finance assistant will speak aloud to a user on a voice call.
Score how natural the answer sounds when heard, from 1 to 5:
5 - sounds like a knowledgeable person talking: short sentences, the main point first, nothing a listener has to see.
4 - natural, with a slightly written phrase or one sentence too many.
3 - understandable aloud but clearly written for reading (lists of figures, parenthetical asides, long sentences).
2 - awkward to hear: symbols, abbreviations or structure that only works on a page.
1 - a document read aloud.
Judge only how it sounds, not whether the finance is right. The phrase "the amount on your screen"
is expected: dollar figures are shown on screen instead of being spoken."""


def _judge(question: str, spoken: str) -> Naturalness:
    from langchain_anthropic import ChatAnthropic

    from src.core.config import get_settings

    judge = ChatAnthropic(
        model=get_settings().voice.classifier_model,  # Haiku: cheap enough to judge every answer
        anthropic_api_key=get_settings().anthropic_api_key,
        max_tokens=300,
    ).with_structured_output(Naturalness)
    return judge.invoke([
        {"role": "system", "content": _JUDGE_RUBRIC},
        {"role": "user", "content": f"The user asked: {question}\n\nThe assistant will say:\n{spoken}"},
    ])


def _spoken(answer: str) -> str:
    """What the worker would say for this answer, in discreet mode."""
    import asyncio

    from src.voice.turns import END, TurnOutcome, spoken_answer

    async def run() -> str:
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(answer)
        queue.put_nowait(END)
        said = [s async for s in spoken_answer(queue.get, TurnOutcome(), notice_s=5, abandon_s=10, discreet=True)]
        return " ".join(" ".join(said).split())

    return asyncio.run(run())


class TestVoiceAnswers:
    results: list[dict[str, Any]] = []

    @pytest.mark.parametrize("query,expected,profile", ANSWER_CASES, ids=[a.value for _, a, _ in ANSWER_CASES])
    def test_answer_is_short_plain_and_discreet(self, query, expected, profile):
        from src.agents.base_agent import _DISCLAIMER
        from src.workflow.graph import run_workflow

        result = run_workflow(query, user_profile=profile, channel="voice")
        assert result["agent_used"] == expected.value
        body = result["final_response"].replace(_DISCLAIMER, "").strip()
        words = len(body.split())
        spoken = _spoken(result["final_response"])
        verdict = _judge(query, spoken)
        self.results.append({"agent": expected.value, "words": words, "score": verdict.score})
        print(f"\n[{expected.value}] {words} words, natural {verdict.score}/5 ({verdict.reason})\n  {spoken}")

        assert words <= MAX_WORDS, body
        assert not _MARKDOWN.search(body), body
        assert not _AMOUNT.search(body), f"dollar amount in a discreet answer: {body}"
        # The backstop should have had nothing to replace; when it does, the
        # sentence sounds broken ("for every the amount on your screen").
        assert AMOUNT_ON_SCREEN not in spoken, spoken
        assert verdict.score >= 3, verdict.reason

    def test_naturalness_overall(self):
        if len(self.results) < len(ANSWER_CASES):
            pytest.skip("run with the answer cases")
        mean = sum(r["score"] for r in self.results) / len(self.results)
        print(f"\nmean naturalness {mean:.2f}/5 over {len(self.results)} answers")
        assert mean >= 4.0
