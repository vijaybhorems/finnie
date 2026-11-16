"""src/voice/limits.py: call length, idle hang-up, turns and the daily budget."""
from __future__ import annotations

from src.core.config import VoiceConfig
from src.voice.limits import Action, CallMonitor, call_budget_seconds

VOICE = VoiceConfig()  # 20 min, warn at 18, idle 90/120 s, 60 turns, 30 min/day


def _run(monitor: CallMonitor, until: float, start: float = 0.0) -> list[tuple[float, Action]]:
    """Tick once a second; the actions and when they fired."""
    fired, t = [], start
    while t <= until:
        action = monitor.tick(t)
        if action:
            fired.append((t, action))
        t += 1
    return fired


def _talking(monitor: CallMonitor, every: float, until: float) -> list[tuple[float, Action]]:
    """A user who speaks for a second every `every` seconds."""
    fired, t = [], 0.0
    while t <= until:
        if t % every == 0:
            monitor.set_user_speaking(True, t)
        elif t % every == 1:
            monitor.set_user_speaking(False, t)
        action = monitor.tick(t)
        if action:
            fired.append((t, action))
        t += 1
    return fired


class TestBudget:
    def test_full_call_when_the_day_is_fresh(self):
        assert call_budget_seconds(VOICE, 0) == 20 * 60

    def test_capped_by_what_is_left_today(self):
        assert call_budget_seconds(VOICE, 25 * 60) == 5 * 60
        assert call_budget_seconds(VOICE, 31 * 60) == 0


class TestCallLength:
    def test_warns_at_18_and_ends_at_20_minutes(self):
        fired = _talking(CallMonitor(VOICE, 0, now=0), every=30, until=21 * 60)
        assert fired == [(18 * 60, Action.WARN_TIME), (20 * 60, Action.END_TIME)]

    def test_daily_budget_shortens_the_call_and_moves_the_warning(self):
        fired = _talking(CallMonitor(VOICE, 25 * 60, now=0), every=30, until=6 * 60)
        assert fired == [(3 * 60, Action.WARN_TIME), (5 * 60, Action.END_TIME)]

    def test_no_warning_when_the_call_is_shorter_than_the_lead(self):
        fired = _talking(CallMonitor(VOICE, 29 * 60, now=0), every=30, until=2 * 60)
        assert fired == [(60, Action.END_TIME)]

    def test_time_runs_out_even_while_finnie_is_speaking(self):
        monitor = CallMonitor(VOICE, 0, now=0)
        monitor.set_agent_busy(True, 0)
        assert monitor.tick(20 * 60) == Action.END_TIME

    def test_nothing_after_the_call_ends(self):
        monitor = CallMonitor(VOICE, 0, now=0)
        assert monitor.tick(20 * 60) == Action.END_TIME
        assert _run(monitor, until=30 * 60, start=20 * 60 + 1) == []


class TestIdle:
    def test_prompt_at_90_then_hang_up_at_120_seconds(self):
        assert _run(CallMonitor(VOICE, 0, now=0), until=130) == [(90, Action.PROMPT_IDLE), (120, Action.END_IDLE)]

    def test_saying_the_prompt_does_not_restart_the_clock(self):
        monitor = CallMonitor(VOICE, 0, now=0)
        assert monitor.tick(90) == Action.PROMPT_IDLE
        monitor.set_agent_busy(True, 90)
        monitor.set_agent_busy(False, 92)
        assert _run(monitor, until=125, start=93) == [(120, Action.END_IDLE)]

    def test_answering_the_prompt_keeps_the_call(self):
        monitor = CallMonitor(VOICE, 0, now=0)
        assert monitor.tick(90) == Action.PROMPT_IDLE
        monitor.set_user_speaking(True, 100)
        monitor.set_user_speaking(False, 101)
        # A fresh idle period: prompted again 90 s after they stopped speaking.
        assert _run(monitor, until=230, start=102) == [(191, Action.PROMPT_IDLE), (221, Action.END_IDLE)]

    def test_a_long_answer_is_not_idle(self):
        monitor = CallMonitor(VOICE, 0, now=0)
        monitor.set_user_speaking(True, 0)
        monitor.set_user_speaking(False, 2)
        monitor.set_agent_busy(True, 2)
        assert _run(monitor, until=200, start=3) == []  # Finnie working the whole time
        monitor.set_agent_busy(False, 200)
        assert monitor.tick(289) is None and monitor.tick(290) == Action.PROMPT_IDLE


class TestTurns:
    def test_sixty_turns_then_the_limit(self):
        monitor = CallMonitor(VOICE, 0, now=0)
        assert all(monitor.start_turn() for _ in range(60))
        assert not monitor.turn_limit_reached
        assert monitor.start_turn() is False
        assert monitor.turn_limit_reached
        assert monitor.tick(1) is None  # the worker ends the call once the goodbye is said
