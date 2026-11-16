"""Simulated calls (design doc, Evaluation layer 5): scenarios and the checks on them.

A scenario is what a caller does on a call: speak (with pauses, noise, an
accent or speed), cut in while Finnie talks, say "uh-huh", ask an LLM-written
follow-up, or stay silent. tests/evals/sim_caller.py plays it over audio against
the real worker; this module only describes the scenarios and judges what the
call left behind, so CI can test the checks without LiveKit.

What a call leaves behind (`Observed`):
  turns      the worker's local turn log (VOICE_TURN_LOG): per answered turn, the
             question as transcribed, what was spoken, the agent, interrupted
  summary    the worker's `voice_call_summary` line: graph runs (`turns`),
             interruptions, why the call ended, cost
  answers    the page's written answers (the `answer` events), with amounts
  transcript LiveKit's transcription stream, both sides
  ended_by_worker, first_audio_s (the caller's own clock)

Checks are deterministic. The one judgement call, how natural the answers sound
aloud, is left to the Haiku judge in test_simulated_calls.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union

from src.voice.limits import IDLE_PROMPT
from src.voice.speech import AMOUNT_ON_SCREEN
from src.voice.turns import SECURITIES_LINE

Part = Union[str, float]  # words to say, or seconds of silence mid-utterance


@dataclass(frozen=True)
class Speak:
    """Say `parts` (text and pauses), then wait.

    wait="answer": until Finnie has finished replying.
    wait="speaking": until Finnie starts replying, then `then_s` more seconds;
                     the next action cuts in.
    wait="none": go straight on.
    """
    parts: tuple[Part, ...]
    voice: str = "Samantha"
    rate: int = 175
    noise: bool = False
    wait: Literal["answer", "speaking", "none"] = "answer"
    then_s: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(p for p in self.parts if isinstance(p, str))


@dataclass(frozen=True)
class FollowUp:
    """An LLM-written next line, from the call so far and this goal; then wait for the answer."""
    goal: str
    voice: str = "Samantha"


@dataclass(frozen=True)
class Silence:
    """Say nothing for up to `seconds`, or until the worker ends the call."""
    seconds: float


Action = Union[Speak, FollowUp, Silence]


@dataclass
class Observed:
    turns: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    answers: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[tuple[str, str]] = field(default_factory=list)
    ended_by_worker: bool = False
    first_audio_s: list[Optional[float]] = field(default_factory=list)
    said: list[str] = field(default_factory=list)  # the caller's lines, follow-ups included

    def answered(self) -> list[dict[str, Any]]:
        return [t for t in self.turns if t.get("agent") not in (None, "error")]

    def last(self) -> dict[str, Any]:
        return self.turns[-1] if self.turns else {}

    def finnie_said(self) -> str:
        return " ".join(text for who, text in self.transcript if who == "Finnie")


Check = Callable[[Observed], Optional[str]]  # None when it passes, else what went wrong


def _words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9$%/' ]+", " ", (text or "").lower()).split())


def graph_runs(n: int) -> Check:
    """The worker answered exactly `n` turns: more means a question was cut in two."""
    def check(o: Observed) -> Optional[str]:
        runs = o.summary.get("turns")
        return None if runs == n else f"{runs} graph runs, expected {n} (questions: {[t.get('question') for t in o.turns]})"
    check.__name__ = f"graph_runs({n})"
    return check


def heard(*any_of_each: tuple[str, ...] | str) -> Check:
    """The last answered question was transcribed with each group's word (any spelling in the group)."""
    groups = [(g,) if isinstance(g, str) else g for g in any_of_each]

    def check(o: Observed) -> Optional[str]:
        answered = o.answered()
        question = _words(answered[-1].get("question", "")) if answered else ""
        missing = [g for g in groups if not any(_words(w) in question for w in g)]
        return None if not missing else f"question {question!r} is missing {missing}"
    check.__name__ = "heard(" + ", ".join("|".join(g) for g in groups) + ")"
    return check


def agent_in(*agents: str) -> Check:
    def check(o: Observed) -> Optional[str]:
        agent = o.last().get("agent")
        return None if agent in agents else f"last turn went to {agent}, expected {agents}"
    check.__name__ = f"agent_in({', '.join(agents)})"
    return check


def interruptions(low: int, high: Optional[int] = None) -> Check:
    def check(o: Observed) -> Optional[str]:
        n = o.summary.get("interruptions")
        ok = isinstance(n, int) and n >= low and (high is None or n <= high)
        return None if ok else f"{n} interruptions, expected {low}..{'' if high is None else high}"
    check.__name__ = f"interruptions({low}..{'' if high is None else high})"
    return check


def spoken_clean(o: Observed) -> Optional[str]:
    """Nothing for the eye reached the voice: markdown, the written disclaimer, symbols."""
    bad = []
    for turn in o.answered():
        spoken = turn.get("spoken") or ""
        for mark in ("**", "#", "|", "```", "Disclaimer", "http"):
            if mark in spoken:
                bad.append((mark, spoken[:80]))
    return None if not bad else f"spoken text has {bad}"


def securities_line_first(o: Observed) -> Optional[str]:
    """Said before the answer: it is the turn's preamble (spoken first, uninterruptible), not in `spoken`."""
    turn = o.last()
    if SECURITIES_LINE not in (turn.get("preamble") or []):
        return f"preamble was {turn.get('preamble')!r}"
    if SECURITIES_LINE in (turn.get("spoken") or ""):
        return "the securities line was also said inside the answer"
    return None


def mentions(*words: str) -> Check:
    def check(o: Observed) -> Optional[str]:
        spoken = _words(o.last().get("spoken", ""))
        missing = [w for w in words if _words(w) not in spoken]
        return None if not missing else f"spoken answer never says {missing}"
    check.__name__ = f"mentions({', '.join(words)})"
    return check


_SPOKEN_FIGURE = re.compile(r"\$\s?\d|\b\d[\d,.]*\s*(?:dollars|bucks|k\b|thousand dollars|million dollars)", re.I)


def amounts_kept_off_audio(o: Observed) -> Optional[str]:
    """Discreet mode: no dollar figure said aloud, and "it's on your screen" only when it is.

    The page shows the voice answer's own text, which discreet mode also keeps
    free of figures, so pointing the caller at the screen promises what isn't there.
    """
    spoken = o.last().get("spoken") or ""
    if _SPOKEN_FIGURE.search(spoken):
        return f"a dollar figure was spoken: {spoken[:120]!r}"
    written = " ".join(a.get("answer") or "" for a in o.answers)
    if "on your screen" in spoken.lower() and not re.search(r"\$\s?\d", written):
        return f"said the figure is on screen, but the written answer has none: {spoken[:160]!r}"
    return None


def ended(reason: str) -> Check:
    def check(o: Observed) -> Optional[str]:
        got = o.summary.get("reason")
        if not o.ended_by_worker:
            return "the worker didn't end the call"
        return None if got == reason else f"ended with {got!r}, expected {reason!r}"
    check.__name__ = f"ended({reason})"
    return check


def asked_if_still_there(o: Observed) -> Optional[str]:
    return None if IDLE_PROMPT.lower() in o.finnie_said().lower() else "never asked if the caller was still there"


def all_answered(o: Observed) -> Optional[str]:
    bad = [t.get("agent") for t in o.turns if t.get("agent") in (None, "error")]
    return None if not bad and o.turns else f"unanswered turns: {bad or 'no turns at all'}"


@dataclass(frozen=True)
class Scenario:
    name: str
    actions: tuple[Action, ...]
    checks: tuple[Check, ...]
    about: str = ""
    turn_taking: bool = False  # part of the turn-detection sweep (scripts/voice_turn_sweep.py)


def say(*parts: Part, **kw: Any) -> Speak:
    return Speak(tuple(parts), **kw)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "basic", (say("What is an index fund?"),),
        (graph_runs(1), agent_in("finance_qa"), heard("index fund"), spoken_clean, interruptions(0, 0)),
        "One plain question: the baseline for time to first audio.",
    ),
    Scenario(
        "pause_mid_question",
        (say("I've got about", 0.9, "um", 0.7, "forty thousand dollars in my four oh one k. Should I roll it into an IRA?"),),
        (graph_runs(1), heard(("40,000", "forty thousand", "40000"), ("ira",), ("roll",))),
        "A hesitation mid-sentence must not end the turn.", turn_taking=True,
    ),
    Scenario(
        "long_pause_mid_question",
        (say("What's the difference between", 1.6, "a Roth IRA and a traditional IRA?"),),
        (graph_runs(1), heard("roth", "traditional")),
        "A long pause after an unfinished phrase.", turn_taking=True,
    ),
    Scenario(
        "backchannel",
        (say("Can you explain how compound interest works?", wait="speaking", then_s=4.0),
         say("uh huh", wait="answer")),
        (graph_runs(1), interruptions(0, 0), heard("compound interest")),
        "\"Uh-huh\" while Finnie talks is not an interruption.", turn_taking=True,
    ),
    Scenario(
        "interruption",
        (say("Explain what a bond ladder is.", wait="speaking", then_s=3.0),
         say("Actually, what's a certificate of deposit?")),
        (graph_runs(2), interruptions(1), heard("certificate of deposit"), agent_in("finance_qa"), all_answered),
        "A real cut-in stops the answer and the new question is answered.", turn_taking=True,
    ),
    Scenario(
        "one_word_cut_in",
        (say("Explain how a mortgage works.", wait="speaking", then_s=3.0),
         say("Stop.")),
        (interruptions(1),),
        "A lone \"Stop.\" cuts in too: what interruption_min_words above 1 gives up.", turn_taking=True,
    ),
    Scenario(
        "off_topic", (say("Tell me a joke about cats."),),
        (graph_runs(1), agent_in("out_of_scope")),
        "Refused politely; the call carries on.",
    ),
    Scenario(
        "stock_question", (say("How is Apple doing today?"),),
        (agent_in("market_analysis"), securities_line_first, mentions("apple"), spoken_clean),
        "Company name to ticker; the securities line is said before the answer.",
    ),
    Scenario(
        "discreet_amounts",
        (say("If I invest ten thousand dollars at seven percent for ten years, how much will I have?"),),
        (all_answered, amounts_kept_off_audio),
        "Dollar figures stay on screen unless the caller turns amounts on.",
    ),
    Scenario(
        "noisy_room", (say("What is a Roth IRA?", noise=True),),
        (all_answered, heard("roth"), agent_in("finance_qa", "tax_education")),
        "Background noise.",
    ),
    Scenario(
        "fast_uk_speaker", (say("What's the difference between a stock and a bond?", voice="Daniel", rate=235),),
        (all_answered, heard("stock", "bond")),
        "A fast British speaker.",
    ),
    Scenario(
        "indian_english", (say("Should I pay off my credit card before I start investing?", voice="Rishi"),),
        (all_answered, heard("credit card", ("invest", "investing"))),
        "An Indian English speaker.",
    ),
    Scenario(
        "follow_up",
        (say("What is a Roth IRA?"),
         FollowUp("Ask one natural follow-up about something specific Finnie just said, the way a person "
                  "would on the phone. One short sentence, no greeting.")),
        (graph_runs(2), all_answered),
        "An LLM caller keeps the conversation going.",
    ),
    Scenario(
        "idle", (say("What is a dividend?"), Silence(90)),
        (all_answered, asked_if_still_there, ended("end_idle")),
        "Silence: \"Are you still there?\", then a hang-up. Needs the short idle times the eval sets.",
    ),
)

# Short idle times so the idle scenario takes under a minute; long enough that
# no other scenario sits quiet that long.
EVAL_WORKER_SETTINGS = {"idle_prompt_seconds": 15, "idle_hangup_seconds": 25}


def run_checks(scenario: Scenario, observed: Observed) -> dict[str, Optional[str]]:
    results = {}
    for check in scenario.checks:
        try:
            results[check.__name__] = check(observed)
        except Exception as exc:  # noqa: BLE001 — a broken check is a failure, not a crash
            results[check.__name__] = f"check raised {type(exc).__name__}: {exc}"
    return results


def by_name(names: Optional[list[str]] = None) -> list[Scenario]:
    if not names:
        return list(SCENARIOS)
    known = {s.name: s for s in SCENARIOS}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}; known: {sorted(known)}")
    return [known[n] for n in names]
