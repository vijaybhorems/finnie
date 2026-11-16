"""Speech shaping: written answers as they would be spoken (design doc, Evaluation layer 1).

Free and in CI. Each case is an answer in the shape an agent writes it
(markdown, tables, tickers, amounts, the written disclaimer), streamed in
small pieces through the same path the voice worker uses
(`turns.spoken_answer`), in discreet mode and with amounts turned on.
Checks are deterministic: nothing here is judged by a model.
"""
from __future__ import annotations

import asyncio
import re

import pytest

from src.agents.base_agent import _DISCLAIMER
from src.voice.speech import AMOUNT_ON_SCREEN, SESSION_DISCLAIMER, TICKER_NAMES
from src.voice.turns import END, SECURITIES_LINE, TurnOutcome, preamble, spoken_answer

# Written the way each agent writes, disclaimer included where agents add it.
ANSWERS = {
    "finance_qa": (
        "## What a P/E ratio tells you\n\n"
        "The **price-to-earnings (P/E) ratio** compares a share price to earnings per share.\n\n"
        "- A P/E of 20 means investors pay $20 for each $1 of yearly earnings.\n"
        "- The S&P 500 has averaged ~16 over the long run.\n\n"
        "Compare it within an industry, e.g. banks vs. banks."
    ),
    "portfolio": (
        "Your portfolio is worth **$48,250.75**, up 6.2% this year.\n\n"
        "| Holding | Value | Weight |\n|---|---|---|\n"
        "| AAPL | $21,400 | 44% |\n| MSFT | $15,100 | 31% |\n| VTI | $11,750 | 25% |\n\n"
        "Tech is about 75% of it, so a sector drop would hit you hard." + _DISCLAIMER
    ),
    "market_analysis": (
        "**NVDA** closed at $131.20, up 2.4%, and **TSLA** fell 1.1%.\n\n"
        "1. Nvidia (NVDA) is above its 50-day average.\n"
        "2. The RSI is 64: strong, not yet overbought.\n\n"
        "SPY rose 0.5% and QQQ 0.8%." + _DISCLAIMER
    ),
    "goal_planning": (
        "To retire at 55 you need about **$1.2M**, or $48k a year at a 4% withdrawal rate.\n\n"
        "Saving $2,500 a month at 7% gets you there in about 19 years. Max your 401(k) first, "
        "then a Roth IRA."
    ),
    "tax_education": (
        "### Roth vs traditional\n\n"
        "* **Roth IRA**: pay tax now, withdraw tax-free.\n"
        "* **Traditional 401(k)**: deduct now, pay tax later.\n\n"
        "The 2025 401k limit is $23,500; a 403(b) has the same limit. See [IRS](https://irs.gov)."
    ),
    "news_synthesizer": (
        "**Markets today:** stocks rose after the Fed held rates. Apple (AAPL) gained 1.8% on "
        "earnings, and JPM said loan demand was ~3% higher YoY.\n\n> \"Inflation is easing,\" "
        "the Fed chair said." + _DISCLAIMER
    ),
}
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*+]\s|\]\(|```|\|)", re.MULTILINE)
_AMOUNT = re.compile(r"\$\s?\d|\b\d[\d,.]*\s*(?:thousand|million|billion)?\s*dollars\b", re.IGNORECASE)


def _pieces(text: str, size: int = 7) -> list[str]:
    """Token-sized pieces, so markers and amounts split across them as they do live."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def _speak(answer: str, discreet: bool) -> str:
    async def run() -> str:
        queue: asyncio.Queue = asyncio.Queue()
        for piece in _pieces(answer):
            queue.put_nowait(piece)
        queue.put_nowait(END)
        said = [s async for s in spoken_answer(
            queue.get, TurnOutcome(), notice_s=5, abandon_s=10, discreet=discreet,
        )]
        return " ".join(" ".join(said).split())

    return asyncio.run(run())


@pytest.mark.parametrize("agent", ANSWERS)
@pytest.mark.parametrize("discreet", [True, False], ids=["discreet", "amounts_on"])
class TestEveryAnswerIsSpeakable:
    def test_no_markdown_or_written_disclaimer(self, agent, discreet):
        spoken = _speak(ANSWERS[agent], discreet)
        assert not _MARKDOWN.search(spoken), spoken
        assert "Disclaimer" not in spoken and "not financial advice" not in spoken.lower()
        assert spoken  # something is still said

    def test_tickers_and_account_names_are_said_as_words(self, agent, discreet):
        spoken = _speak(ANSWERS[agent], discreet)
        assert "401(k)" not in spoken and "401k" not in spoken and "403(b)" not in spoken
        for symbol in TICKER_NAMES:
            if len(symbol) > 2:
                assert not re.search(rf"\b{re.escape(symbol)}\b", spoken), (symbol, spoken)
        assert "S&P" not in spoken and "P/E" not in spoken

    def test_amounts_follow_the_users_choice(self, agent, discreet):
        spoken = _speak(ANSWERS[agent], discreet)
        had_amounts = bool(_AMOUNT.search(ANSWERS[agent]))
        if discreet:
            assert not _AMOUNT.search(spoken), spoken
            assert (AMOUNT_ON_SCREEN in spoken) == had_amounts
        else:
            assert AMOUNT_ON_SCREEN not in spoken
            assert "$1.2M" not in spoken and "$48k" not in spoken  # said as words


def test_percentages_survive_discreet_mode():
    spoken = _speak(ANSWERS["portfolio"], discreet=True)
    assert "6.2%" in spoken and "75%" in spoken


def test_amounts_on_reads_suffixes_as_words():
    spoken = _speak(ANSWERS["goal_planning"], discreet=False)
    assert "1.2 million dollars" in spoken and "48 thousand dollars" in spoken


class TestSpokenDisclaimers:
    """Design doc, Open question 3: one line per session, one before specific securities."""

    def test_wording(self):
        assert SESSION_DISCLAIMER == "I offer general financial education, not personal advice."
        assert SECURITIES_LINE == "This isn't a recommendation to buy or sell."

    @pytest.mark.parametrize("agent,tickers,expected", [
        ("market_analysis", [], True),
        ("portfolio", [], True),
        ("news_synthesizer", ["AAPL"], True),
        ("news_synthesizer", [], False),
        ("finance_qa", ["AAPL"], False),
        ("tax_education", [], False),
        ("goal_planning", [], False),
    ])
    def test_securities_line_fires_in_code(self, agent, tickers, expected):
        from types import SimpleNamespace

        lines = preamble(SimpleNamespace(on_topic=True, agent=agent, tickers=tickers), filler=False)
        assert (SECURITIES_LINE in lines) is expected
