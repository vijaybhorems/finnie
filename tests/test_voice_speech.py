"""Speech shaping for the voice spike (design doc, Evaluation layer 1)."""
from __future__ import annotations

import pytest

from src.voice.speech import AMOUNT_ON_SCREEN, SpeechBuffer, for_speech, mask_amounts, strip_written_disclaimer
from src.agents.base_agent import _DISCLAIMER


def test_written_disclaimer_is_not_spoken():
    answer = "A Roth IRA is funded with after-tax money." + _DISCLAIMER
    assert strip_written_disclaimer(answer) == "A Roth IRA is funded with after-tax money."


def test_disclaimer_chunk_alone_becomes_empty():
    # stream_workflow flushes the disclaimer as its own final chunk.
    assert strip_written_disclaimer(_DISCLAIMER).strip() == ""


def test_markdown_is_removed():
    text = "## Roth vs traditional\n- **Roth:** pay tax now\n- *Traditional:* pay tax later\n[IRS](https://irs.gov)"
    spoken = for_speech(text)
    for marker in ("#", "**", "*", "- ", "](", "https"):
        assert marker not in spoken
    assert "Roth: pay tax now" in spoken and "IRS" in spoken


def test_buffer_releases_whole_sentences_with_markers_split_across_tokens():
    buffer = SpeechBuffer()
    out = [buffer.push(t) for t in ["A **Ro", "th IRA** grows ", "tax-free. Then", " more"]]
    assert out[:2] == ["", ""]
    assert out[2] == "A Roth IRA grows tax-free."
    assert buffer.flush() == " Then more"


def test_plain_numbers_and_tickers_pass_through():
    assert for_speech("AAPL is up 2.5% to $190.") == "AAPL is up 2.5% to $190."


class TestMaskAmounts:
    """Discreet mode's backstop: dollar figures never reach text-to-speech."""

    @pytest.mark.parametrize("text", [
        "You have $12,345.67 in cash.",
        "It's worth $1.2M now.",
        "A $3 million portfolio.",
        "About $40k a year.",
        "That's 2,000 dollars a month.",
        "Roughly 1.5 million dollars.",
        "USD 500 in fees.",
        "Save two thousand dollars a month.",
        "A few hundred dollars.",
        "Fifty-five dollars a share.",
        "£500 or €200.",
    ])
    def test_amounts_are_replaced(self, text):
        spoken = mask_amounts(text)
        assert AMOUNT_ON_SCREEN in spoken
        assert not any(ch.isdigit() for ch in spoken.replace("401", ""))
        assert "dollar" not in spoken.lower()

    @pytest.mark.parametrize("text", [
        "Up 6.5% this year, mostly in tech.",
        "Your 401(k) and a Roth IRA.",
        "$AAPL rose three percent.",
        "Two or three funds, held for 10 years.",
        "In 2028 you could retire.",
    ])
    def test_everything_else_is_untouched(self, text):
        assert mask_amounts(text) == text

    def test_a_range_is_said_once(self):
        assert mask_amounts("between $100 and $200") == f"between {AMOUNT_ON_SCREEN}"
        assert mask_amounts("$5k-$10k") == AMOUNT_ON_SCREEN

    def test_no_rounding_words_before_a_hidden_figure(self):
        assert mask_amounts("Save about $2,000 a month.") == f"Save {AMOUNT_ON_SCREEN} a month."
