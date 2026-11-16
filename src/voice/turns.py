"""What the worker says around an answer, and when a turn goes wrong.

Pure (no LiveKit), so CI tests it. The worker wires it to LiveKit events.

- `preamble()`: lines said before the answer: the filler on slow routes, and
  the securities line when the turn is about specific securities. Said first
  so an interruption can't cut them off (design doc, Open question 3).
- `spoken_answer()`: turns the graph's token stream into speech, with the
  slow-turn notice at `slow_turn_notice_seconds`, an apology and a stop at
  `slow_turn_abandon_seconds` with no first token, and an apology if the
  graph fails. With `discreet=True` (the default for voice) dollar figures
  are masked (`speech.mask_amounts`).
- `StallWatch`: notices speech-to-text going quiet (the user spoke, nothing
  was transcribed) and says when to warn and when to end the call.
- `heard_answer()`: what the user heard of an answer, minus the preamble, for
  `interrupted_after` on the next turn.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from src.voice.prefetch import FILLERS
from src.voice.speech import SpeechBuffer, mask_amounts, strip_written_disclaimer

SECURITIES_LINE = "This isn't a recommendation to buy or sell."
SLOW_NOTICE = "This is taking longer than usual."
SORRY = "Sorry, I couldn't answer that. Try again, or ask in the chat."
HEARING_TROUBLE = "I'm having trouble hearing you. You can keep typing in the chat."
HEARING_GIVE_UP = "I still can't hear you, so I'll end the call here. You can keep typing in the chat."

# Lines the worker sends to text-to-speech at once rather than waiting for
# the next sentence to fill the buffer.
STANDALONE_LINES = frozenset({SECURITIES_LINE, SLOW_NOTICE, SORRY, *FILLERS.values()})

# Routes whose answers are about specific securities. Portfolio answers are
# about the user's own holdings; market answers name funds even for "how's the
# market" (the SPY/QQQ snapshot); news only when the question names a company.
_ALWAYS_SECURITIES = {"market_analysis", "portfolio"}
_SECURITIES_WITH_TICKER = {"news_synthesizer"}

END = object()  # the graph finished


def needs_securities_line(agent: str, tickers: list[str]) -> bool:
    return agent in _ALWAYS_SECURITIES or (agent in _SECURITIES_WITH_TICKER and bool(tickers))


def preamble(verdict: Any, filler: bool = True) -> list[str]:
    """Lines to say before the answer, given the classifier's verdict (or None)."""
    if verdict is None or not getattr(verdict, "on_topic", False):
        return []
    agent = getattr(verdict, "agent", "")
    lines = []
    if filler and agent in FILLERS:
        lines.append(FILLERS[agent])
    if needs_securities_line(agent, list(getattr(verdict, "tickers", None) or [])):
        lines.append(SECURITIES_LINE)
    return lines


def heard_answer(spoken: str) -> str:
    """What was heard of the answer itself: the preamble lines removed."""
    text = " ".join((spoken or "").split())
    changed = True
    while changed:
        changed = False
        for line in STANDALONE_LINES:
            if text.startswith(line):
                text, changed = text[len(line):].lstrip(), True
    return text


@dataclass
class TurnOutcome:
    completed: bool = False   # the whole answer was handed to TTS
    abandoned: bool = False   # no first token in time; SORRY said instead
    failed: bool = False      # the graph raised; SORRY said instead
    first_token_s: Optional[float] = None


async def spoken_answer(
    next_item: Callable[[], Awaitable[Any]],
    outcome: TurnOutcome,
    *,
    notice_s: float,
    abandon_s: float,
    failed: Callable[[], bool] = lambda: False,
    discreet: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Yield what to say for one turn. `next_item()` returns a token or END.

    `failed()` is checked on each token: the graph reports an error as one
    final token (its error text), which is never spoken. `discreet` masks
    dollar figures in the answer.
    """
    shape = mask_amounts if discreet else (lambda text: text)
    start = clock()
    buffer = SpeechBuffer()
    noticed = False
    while True:
        if outcome.first_token_s is None:
            limit = abandon_s if noticed else notice_s
            try:
                item = await asyncio.wait_for(next_item(), timeout=max(0.0, limit - (clock() - start)))
            except asyncio.TimeoutError:
                if not noticed:
                    noticed = True
                    yield SLOW_NOTICE
                    continue
                outcome.abandoned = True
                yield SORRY
                return
        else:
            item = await next_item()
        if item is END:
            tail = shape(buffer.flush())
            if tail.strip():
                yield tail
            outcome.completed = True
            return
        if failed():
            outcome.failed = True
            yield SORRY
            return
        if outcome.first_token_s is None:
            outcome.first_token_s = round(clock() - start, 3)
        spoken = shape(buffer.push(strip_written_disclaimer(item)))
        if spoken.strip():
            yield spoken


class StallWatch:
    """Speech-to-text stalls: the user spoke, and no words came back in time.

    Two kinds, counted together:
    - a quiet stall: real speech (at least `min_speech_s`, so a cough doesn't
      count) and no transcript, not even an interim one, within `stall_s` of
      the end of speech;
    - an outage: speech-to-text errors. LiveKit retries several times a second,
      so an outage counts once when it starts, and again each `stall_s` it
      lasts without a transcript.

    Any transcript resets the count, so only stalls in a row end the call:
    `tick()` says PROMPT (say HEARING_TROUBLE) until the `limit`-th, then END
    (say HEARING_GIVE_UP and end the call).
    """

    PROMPT = "prompt"
    END = "end"

    def __init__(self, stall_s: float, limit: int, min_speech_s: float = 0.6) -> None:
        self._stall_s, self._limit, self._min_speech_s = stall_s, limit, min_speech_s
        self.in_a_row = 0
        self._speech_started: Optional[float] = None
        self._heard = False
        self._waiting_since: Optional[float] = None  # quiet stall: speech ended, no words yet
        self._outage_due: Optional[float] = None     # outage: when it next counts

    def speech_started(self, now: float) -> None:
        self._speech_started, self._heard, self._waiting_since = now, False, None

    def speech_ended(self, now: float) -> None:
        if self._speech_started is None:
            return
        long_enough = now - self._speech_started >= self._min_speech_s
        self._speech_started = None
        if long_enough and not self._heard:
            self._waiting_since = now

    def transcript(self) -> None:
        self._heard, self.in_a_row = True, 0
        self._waiting_since = self._outage_due = None

    def failure(self, now: float) -> None:
        """A speech-to-text error. Retries during one outage don't count again."""
        if self._outage_due is None:
            self._outage_due = now

    def tick(self, now: float) -> Optional[str]:
        if self._outage_due is not None and now >= self._outage_due:
            self._outage_due = now + self._stall_s  # still down by then: counts again
            self._waiting_since = None
            return self._count()
        if self._waiting_since is None or now - self._waiting_since < self._stall_s:
            return None
        self._waiting_since = None
        return self._count()

    def _count(self) -> str:
        self.in_a_row += 1
        return self.END if self.in_a_row >= self._limit else self.PROMPT
