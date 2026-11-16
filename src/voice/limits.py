"""Limits during a voice call: length, idle time, turns, and the daily budget.

Pure bookkeeping, no LiveKit: the worker reports what happens (the user or
Finnie starts or stops speaking, a turn starts) and calls `tick()` about once
a second. `tick()` returns at most one action, and each one at most once per
call, except the idle prompt, which can come again after the user speaks.

The call's length budget is the smaller of `voice.max_call_minutes` and what
is left of the user's `voice.daily_minutes_per_user`, so one call never runs
past either. The warning comes the same distance before the end as
`call_warning_minutes` is before `max_call_minutes` (2 minutes by default).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from src.core.config import VoiceConfig


class Action(str, Enum):
    WARN_TIME = "warn_time"      # say TIME_WARNING, keep going
    END_TIME = "end_time"        # say TIME_UP, end the call
    PROMPT_IDLE = "prompt_idle"  # say IDLE_PROMPT, keep going
    END_IDLE = "end_idle"        # say IDLE_END, end the call
    END_TURNS = "end_turns"      # say TURN_LIMIT instead of answering, end the call


# Spoken by the worker.
TIME_WARNING = "Just so you know, we have about two minutes left on this call."
TIME_UP = "That's the time limit for this call. You can keep going in the chat. Goodbye."
IDLE_PROMPT = "Are you still there?"
IDLE_END = "I'll hang up for now. Talk soon."
TURN_LIMIT = "That's as many questions as I can take in one call. You can keep going in the chat. Goodbye."
DAILY_LIMIT = "You've used today's voice time. You can keep going in the chat."

LINES = {
    Action.WARN_TIME: TIME_WARNING,
    Action.END_TIME: TIME_UP,
    Action.PROMPT_IDLE: IDLE_PROMPT,
    Action.END_IDLE: IDLE_END,
    Action.END_TURNS: TURN_LIMIT,
}
ENDS_CALL = {Action.END_TIME, Action.END_IDLE, Action.END_TURNS}


def call_budget_seconds(voice: VoiceConfig, used_today_s: float) -> float:
    """How long this call may run: the per-call cap or what is left of today's."""
    left_today = voice.daily_minutes_per_user * 60 - max(0.0, used_today_s)
    return max(0.0, min(voice.max_call_minutes * 60, left_today))


class CallMonitor:
    def __init__(self, voice: VoiceConfig, used_today_s: float, now: float) -> None:
        self._voice = voice
        self.started = now
        self.budget_s = call_budget_seconds(voice, used_today_s)
        lead_s = max(0.0, (voice.max_call_minutes - voice.call_warning_minutes) * 60)
        # No warning when the whole call is shorter than the warning lead.
        self._warn_at: Optional[float] = self.budget_s - lead_s if self.budget_s > lead_s else None
        self.turns = 0
        self._user_speaking = False
        self._agent_busy = False
        self._quiet_since = now  # nobody speaking and Finnie not working on an answer
        self._prompted = False
        self._fired: set[Action] = set()

    @property
    def agent_busy(self) -> bool:
        return self._agent_busy

    @property
    def turn_limit_reached(self) -> bool:
        return Action.END_TURNS in self._fired

    # ── What happened ───────────────────────────────────────────────────────

    def set_user_speaking(self, speaking: bool, now: float) -> None:
        if speaking:
            self._prompted = False  # they answered "Are you still there?"
        self._user_speaking = speaking
        self._touch(now)

    def set_agent_busy(self, busy: bool, now: float) -> None:
        """Finnie thinking or speaking: the idle clock waits for it.

        Saying IDLE_PROMPT itself doesn't restart the clock, so the call still
        ends `idle_hangup_seconds` after the user last spoke.
        """
        self._agent_busy = busy
        if not self._prompted:
            self._touch(now)

    def start_turn(self) -> bool:
        """Count a user turn; False once the turn limit is passed (answer TURN_LIMIT instead)."""
        self.turns += 1
        if self.turns > self._voice.max_turns_per_call:
            self._fired.add(Action.END_TURNS)
            return False
        return True

    def _touch(self, now: float) -> None:
        if not self._user_speaking and not self._agent_busy:
            self._quiet_since = now

    # ── What to do ──────────────────────────────────────────────────────────

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started)

    def tick(self, now: float) -> Optional[Action]:
        if self._fired & ENDS_CALL:
            return None
        elapsed = self.elapsed(now)
        if elapsed >= self.budget_s:
            return self._fire(Action.END_TIME)
        if self._warn_at is not None and elapsed >= self._warn_at and Action.WARN_TIME not in self._fired:
            return self._fire(Action.WARN_TIME)
        if self._user_speaking or self._agent_busy:
            return None
        quiet = now - self._quiet_since
        if quiet >= self._voice.idle_hangup_seconds:
            return self._fire(Action.END_IDLE)
        if quiet >= self._voice.idle_prompt_seconds and not self._prompted:
            self._prompted = True
            return Action.PROMPT_IDLE
        return None

    def _fire(self, action: Action) -> Action:
        self._fired.add(action)
        return action
