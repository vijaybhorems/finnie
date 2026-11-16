"""src/voice/stt.py: the speech-to-text options and the transcript clean-up."""
from __future__ import annotations

import pytest

from src.voice.stt import KEYTERMS, clean_transcript, deepgram_options


def test_worker_options():
    options = deepgram_options("nova-3")
    assert options["model"] == "nova-3" and options["keyterm"] == KEYTERMS
    assert options["mip_opt_out"] is True and options["smart_format"] is True


@pytest.mark.parametrize("heard,cleaned", [
    # As Deepgram returned them in tests/evals/test_stt_evals.py
    ("My four zero one ks match is 4%.", "My 401(k) match is 4%."),
    ("Can I roll my old four zero one k into an IRA?", "Can I roll my old 401(k) into an IRA?"),
    ("I have a four zero three b through my school district.", "I have a 403(b) through my school district."),
    ("What is a five twenty nine plan?", "What is a 529 plan?"),
    ("Can I use a 5 twenty nine for private school?", "Can I use a 529 for private school?"),
    ("How is the S and P five hundred doing today?", "How is the S&P 500 doing today?"),
    ("Can I use a 5 29 for private school", "Can I use a 529 for private school"),
    ("four oh one k", "401(k)"),
    ("a four fifty seven b", "a 457(b)"),
])
def test_spelled_out_terms(heard, cleaned):
    assert clean_transcript(heard) == cleaned


@pytest.mark.parametrize("text", [
    "I'm forty one and saving four hundred a month.",
    "Five twenty-somethings asked about it.",
    "My 401(k) and the S&P 500.",
])
def test_everything_else_is_untouched(text):
    assert clean_transcript(text) == text
