"""src/voice/turns.py: the preamble, slow and failed turns, and speech-to-text stalls."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.voice.prefetch import FILLERS
from src.voice.speech import AMOUNT_ON_SCREEN
from src.voice.turns import (
    END,
    SECURITIES_LINE,
    SLOW_NOTICE,
    SORRY,
    StallWatch,
    TurnOutcome,
    heard_answer,
    preamble,
    spoken_answer,
)


def _verdict(agent: str, tickers=(), on_topic=True):
    return SimpleNamespace(agent=agent, tickers=list(tickers), on_topic=on_topic)


class TestPreamble:
    @pytest.mark.parametrize("agent,tickers", [
        ("market_analysis", ()), ("market_analysis", ["AAPL"]), ("portfolio", ()), ("news_synthesizer", ["TSLA"]),
    ])
    def test_securities_line_on_specific_securities(self, agent, tickers):
        lines = preamble(_verdict(agent, tickers))
        assert lines[-1] == SECURITIES_LINE

    @pytest.mark.parametrize("agent", ["finance_qa", "tax_education", "goal_planning", "news_synthesizer"])
    def test_no_securities_line_otherwise(self, agent):
        assert SECURITIES_LINE not in preamble(_verdict(agent))

    def test_filler_comes_first_and_can_be_switched_off(self):
        assert preamble(_verdict("market_analysis", ["AAPL"])) == [FILLERS["market_analysis"], SECURITIES_LINE]
        assert preamble(_verdict("market_analysis", ["AAPL"]), filler=False) == [SECURITIES_LINE]

    @pytest.mark.parametrize("agent", ["finance_qa", "tax_education"])
    def test_qa_and_tax_turns_get_a_filler_too(self, agent):
        """Their first sentence lands ~1.3 s after the verdict; the filler is heard first."""
        assert preamble(_verdict(agent)) == [FILLERS[agent]]

    def test_securities_line_never_depends_on_the_filler_switch(self):
        assert preamble(_verdict("portfolio"), filler=False) == [SECURITIES_LINE]

    def test_nothing_for_refusals_or_an_unknown_route(self):
        assert preamble(_verdict("market_analysis", on_topic=False)) == []
        assert preamble(None) == []


def test_heard_answer_drops_the_preamble():
    spoken = f"{FILLERS['market_analysis']} {SECURITIES_LINE} Apple is trading  around"
    assert heard_answer(spoken) == "Apple is trading around"
    assert heard_answer(SECURITIES_LINE) == ""


async def _speak(items, *, notice=0.05, abandon=0.15, failed=lambda: False, delay=0.0, discreet=False):
    """Collect what spoken_answer says for tokens arriving after `delay`."""
    queue: asyncio.Queue = asyncio.Queue()

    async def feed():
        await asyncio.sleep(delay)
        for item in items:
            queue.put_nowait(item)

    asyncio.ensure_future(feed())
    outcome = TurnOutcome()
    said = [s async for s in spoken_answer(
        queue.get, outcome, notice_s=notice, abandon_s=abandon, failed=failed, discreet=discreet,
    )]
    return said, outcome


class TestSpokenAnswer:
    async def test_a_quick_answer_is_spoken_whole(self):
        said, outcome = await _speak(["A P/E ratio compares price to earnings. ", "It is a ratio.", END])
        assert "".join(said).strip() == "A P E ratio compares price to earnings. It is a ratio."  # P/E said as letters
        assert outcome.completed and outcome.first_token_s is not None

    async def test_written_disclaimer_is_not_spoken(self):
        said, _ = await _speak(["Apple is up today.", "\n\n---\n*Disclaimer: not advice.*", END])
        assert "Disclaimer" not in "".join(said)

    async def test_slow_first_token_gets_a_notice(self):
        said, outcome = await _speak(["Here it is.", END], delay=0.1)
        assert said[0] == SLOW_NOTICE and outcome.completed

    async def test_no_first_token_in_time_is_abandoned(self):
        said, outcome = await _speak(["Too late.", END], delay=0.3)
        assert said == [SLOW_NOTICE, SORRY]
        assert outcome.abandoned and not outcome.completed

    async def test_a_long_answer_is_not_cut_off_once_it_started(self):
        queue: asyncio.Queue = asyncio.Queue()

        async def feed():
            queue.put_nowait("First sentence. ")
            await asyncio.sleep(0.3)  # longer than the abandon limit, after the first token
            queue.put_nowait("Second sentence.")
            queue.put_nowait(END)

        asyncio.ensure_future(feed())
        outcome = TurnOutcome()
        said = [s async for s in spoken_answer(queue.get, outcome, notice_s=0.05, abandon_s=0.15)]
        assert "Second sentence." in "".join(said) and SORRY not in said

    async def test_discreet_answers_never_say_an_amount(self):
        # Split across tokens, and in the tail the buffer flushes at the end.
        said, _ = await _speak(["You hold $12,", "345 in cash. ", "About 2,000 dollars", END], discreet=True)
        spoken = "".join(said)
        assert "$" not in spoken and "dollars" not in spoken and spoken.count(AMOUNT_ON_SCREEN) == 2

    async def test_amounts_are_spoken_when_the_user_turned_them_on(self):
        said, _ = await _speak(["You hold $12,345 in cash.", END], discreet=False)
        assert "$12,345" in "".join(said)

    async def test_graph_error_is_an_apology_not_the_error_text(self):
        # The worker's failed() reads sink["agent_used"] == "error", set before the error token.
        said, outcome = await _speak(
            ["I encountered an error: boom. Please try again.", END],
            failed=lambda: True,
        )
        assert said == [SORRY] and outcome.failed


class TestStallWatch:
    def test_speech_with_no_transcript_is_a_stall(self):
        watch = StallWatch(stall_s=3, limit=2)
        watch.speech_started(0)
        watch.speech_ended(2)
        assert watch.tick(4.9) is None
        assert watch.tick(5) == StallWatch.PROMPT

    def test_two_in_a_row_end_the_call(self):
        watch = StallWatch(stall_s=3, limit=2)
        for start, action in ((0, StallWatch.PROMPT), (10, StallWatch.END)):
            watch.speech_started(start)
            watch.speech_ended(start + 2)
            assert watch.tick(start + 5) == action

    def test_a_transcript_resets_the_count(self):
        watch = StallWatch(stall_s=3, limit=2)
        watch.speech_started(0)
        watch.speech_ended(2)
        assert watch.tick(5) == StallWatch.PROMPT
        watch.speech_started(10)
        watch.transcript()
        watch.speech_ended(12)
        assert watch.tick(20) is None and watch.in_a_row == 0

    def test_a_cough_is_not_a_stall(self):
        watch = StallWatch(stall_s=3, limit=2)
        watch.speech_started(0)
        watch.speech_ended(0.3)
        assert watch.tick(10) is None

    def test_an_outage_counts_once_then_again_if_it_lasts(self):
        watch = StallWatch(stall_s=3, limit=2)
        for retry in (1.0, 1.6, 2.2):  # LiveKit's reconnect attempts
            watch.failure(retry)
        assert watch.tick(1.0) == StallWatch.PROMPT
        watch.failure(2.8)
        assert watch.tick(2.9) is None   # the same outage
        assert watch.tick(4.0) == StallWatch.END

    def test_a_transcript_ends_the_outage(self):
        watch = StallWatch(stall_s=3, limit=2)
        watch.failure(0)
        assert watch.tick(0) == StallWatch.PROMPT
        watch.transcript()
        assert watch.tick(10) is None and watch.in_a_row == 0
