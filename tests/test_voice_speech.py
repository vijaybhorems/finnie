"""Speech shaping for the voice spike (design doc, Evaluation layer 1)."""
from __future__ import annotations

from spike.voice.speech import SpeechBuffer, for_speech, strip_written_disclaimer
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
