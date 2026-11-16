"""The simulated-call checks (tests/evals/sim_calls.py) on made-up call results: no LiveKit, no audio."""
from __future__ import annotations

import pytest

from src.voice.limits import IDLE_PROMPT
from src.voice.turns import SECURITIES_LINE
from tests.evals.sim_calls import (
    EVAL_WORKER_SETTINGS,
    SCENARIOS,
    FollowUp,
    Observed,
    Silence,
    Speak,
    agent_in,
    all_answered,
    amounts_kept_off_audio,
    asked_if_still_there,
    by_name,
    ended,
    graph_runs,
    heard,
    interruptions,
    mentions,
    run_checks,
    say,
    securities_line_first,
    spoken_clean,
)


def call(*turns: dict, **kw) -> Observed:
    return Observed(turns=list(turns), **kw)


def turn(question="what is an index fund", agent="finance_qa", spoken="An index fund tracks a market index.", **kw):
    return {"question": question, "agent": agent, "spoken": spoken, **kw}


class TestTurnTaking:
    def test_one_question_one_run_passes(self):
        assert graph_runs(1)(call(turn(), summary={"turns": 1})) is None

    def test_a_question_cut_in_two_fails_and_shows_both_halves(self):
        o = call(turn("i've got about"), turn("forty thousand dollars in my four oh one k"), summary={"turns": 2})
        why = graph_runs(1)(o)
        assert "2 graph runs" in why and "i've got about" in why

    def test_no_summary_fails(self):
        assert graph_runs(1)(call(turn())) is not None

    def test_interruptions_in_range(self):
        assert interruptions(0, 0)(call(summary={"interruptions": 0})) is None
        assert interruptions(1)(call(summary={"interruptions": 3})) is None
        assert interruptions(0, 0)(call(summary={"interruptions": 1})) is not None
        assert interruptions(1)(call(summary={})) is not None


class TestWhatWasHeard:
    def test_any_spelling_in_a_group_counts(self):
        o = call(turn("I've got about $40,000 in my 401(k). Should I roll it into an IRA?"))
        assert heard(("40,000", "forty thousand"), ("ira",), ("roll",))(o) is None

    def test_missing_word_is_named(self):
        why = heard("roth", "traditional")(call(turn("what's the difference between a roth")))
        assert "traditional" in why and "roth" not in why.split("missing")[1]

    def test_uses_the_last_answered_turn_not_an_error(self):
        o = call(turn("what is a roth ira"), turn("garbled", agent="error"))
        assert heard("roth")(o) is None

    def test_agent_of_the_last_turn(self):
        assert agent_in("finance_qa", "tax_education")(call(turn(agent="tax_education"))) is None
        assert "out_of_scope" in agent_in("finance_qa")(call(turn(agent="out_of_scope")))


class TestWhatWasSpoken:
    def test_markdown_or_disclaimer_in_speech_fails(self):
        assert spoken_clean(call(turn())) is None
        assert spoken_clean(call(turn(spoken="**Index funds** are cheap"))) is not None
        assert spoken_clean(call(turn(spoken="Cheap. Disclaimer: not advice"))) is not None

    def test_securities_line_is_the_preamble(self):
        filler = "Let me check the latest market numbers."
        assert securities_line_first(call(turn(spoken="Apple is up today.", preamble=[filler, SECURITIES_LINE]))) is None
        assert "preamble" in securities_line_first(call(turn(spoken="Apple is up today.", preamble=[filler])))
        assert "also said" in securities_line_first(
            call(turn(spoken=f"Apple is up. {SECURITIES_LINE}", preamble=[SECURITIES_LINE])))

    def test_mentions(self):
        assert mentions("apple")(call(turn(spoken="Apple closed higher."))) is None
        assert mentions("apple")(call(turn(spoken="It closed higher."))) is not None

    @pytest.mark.parametrize("spoken", [
        "You'd have about $19,671.",
        "You'd have about 19,671 dollars.",
        "Roughly 20k after ten years.",
        "About 19.7 thousand dollars.",
    ])
    def test_a_spoken_dollar_figure_fails(self, spoken):
        o = call(turn(spoken=spoken), answers=[{"answer": "About $19,671."}])
        assert "spoken" in amounts_kept_off_audio(o)

    def test_figure_on_screen_only_passes(self):
        o = call(turn(spoken="You'd have the amount on your screen, about double at seven percent."),
                 answers=[{"answer": "About **$19,671**."}])
        assert amounts_kept_off_audio(o) is None

    def test_no_figure_and_no_promise_passes(self):
        o = call(turn(spoken="It roughly doubles."), answers=[{"answer": "It roughly doubles."}])
        assert amounts_kept_off_audio(o) is None

    def test_pointing_at_an_empty_screen_fails(self):
        o = call(turn(spoken="It roughly doubles. The exact projection is on your screen."),
                 answers=[{"answer": "It roughly doubles. The exact projection is on your screen."}])
        assert "written answer has none" in amounts_kept_off_audio(o)


class TestCallEnd:
    def test_idle_hang_up(self):
        o = call(turn(), summary={"reason": "end_idle"}, ended_by_worker=True,
                 transcript=[("Finnie", "A dividend is..."), ("Finnie", IDLE_PROMPT)])
        assert ended("end_idle")(o) is None
        assert asked_if_still_there(o) is None

    def test_caller_hung_up_first_fails(self):
        assert "didn't end" in ended("end_idle")(call(summary={"reason": "end_idle"}))

    def test_wrong_reason(self):
        o = call(summary={"reason": "end_max_minutes"}, ended_by_worker=True)
        assert "end_max_minutes" in ended("end_idle")(o)

    def test_never_prompted(self):
        assert asked_if_still_there(call(transcript=[("You", "are you there?")])) is not None

    def test_all_answered(self):
        assert all_answered(call(turn(), turn())) is None
        assert all_answered(call(turn(), turn(agent="error"))) is not None
        assert all_answered(call()) is not None


class TestScenarios:
    def test_names_are_unique_and_every_scenario_has_checks(self):
        names = [s.name for s in SCENARIOS]
        assert len(names) == len(set(names))
        assert all(s.checks and s.actions for s in SCENARIOS)

    def test_actions_are_known_kinds(self):
        for s in SCENARIOS:
            assert all(isinstance(a, (Speak, FollowUp, Silence)) for a in s.actions), s.name

    def test_turn_taking_scenarios_check_runs_or_interruptions(self):
        """The sweep tallies those two checks, so each turn-taking scenario has one."""
        for s in (s for s in SCENARIOS if s.turn_taking):
            assert any(c.__name__.startswith(("graph_runs", "interruptions")) for c in s.checks), s.name

    def test_idle_scenario_outlasts_the_eval_idle_times(self):
        idle = by_name(["idle"])[0]
        silence = next(a for a in idle.actions if isinstance(a, Silence))
        assert silence.seconds > EVAL_WORKER_SETTINGS["idle_hangup_seconds"] + EVAL_WORKER_SETTINGS["idle_prompt_seconds"]

    def test_eval_settings_are_real_voice_settings(self):
        from src.core.config import VoiceConfig

        assert set(EVAL_WORKER_SETTINGS) <= set(VoiceConfig.model_fields)

    def test_by_name(self):
        assert [s.name for s in by_name(["basic", "idle"])] == ["basic", "idle"]
        assert len(by_name(None)) == len(SCENARIOS)
        with pytest.raises(ValueError, match="nope"):
            by_name(["nope"])

    def test_speak_text_skips_pauses(self):
        assert say("I've got about", 0.9, "um", 0.7, "forty").text == "I've got about um forty"

    def test_a_raising_check_is_a_failure_not_a_crash(self):
        def broken(o):
            raise KeyError("x")
        s = by_name(["basic"])[0]
        s = type(s)(s.name, s.actions, (broken,))
        assert "raised KeyError" in run_checks(s, Observed())["broken"]


class TestVoiceSettingsOverride:
    def test_env_json_overrides_voice_settings(self, monkeypatch):
        from src.core.config import get_settings

        monkeypatch.setenv("FINNIE_VOICE_SETTINGS", '{"endpointing_min_delay": 0.8, "idle_prompt_seconds": 12}')
        voice = get_settings().voice
        assert voice.endpointing_min_delay == 0.8 and voice.idle_prompt_seconds == 12

    @pytest.mark.parametrize("raw", ['{"idle_prompt_seconds": "soon"}', '{"endpointing_min_dealy": 0.8}'])
    def test_unknown_or_bad_values_are_refused(self, monkeypatch, raw):
        from src.core.config import get_settings

        monkeypatch.setenv("FINNIE_VOICE_SETTINGS", raw)
        with pytest.raises(ValueError):
            get_settings()
