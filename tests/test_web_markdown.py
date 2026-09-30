"""Tests for escaping model output before Streamlit markdown renders it."""
from src.web_app.markdown import render_safe, render_safe_stream


def test_render_safe_escapes_dollar_signs():
    assert render_safe("$40 and $2") == "\\$40 and \\$2"


def test_render_safe_leaves_text_without_dollars_unchanged():
    assert render_safe("A **P/E ratio** of 20") == "A **P/E ratio** of 20"


def test_render_safe_stream_escapes_each_chunk():
    chunks = ["A stock at $", "40 per share earned ", "$2 per share", ""]
    out = list(render_safe_stream(iter(chunks)))
    assert out == ["A stock at \\$", "40 per share earned ", "\\$2 per share", ""]
    assert "".join(out) == render_safe("".join(chunks))


def test_render_safe_stream_is_lazy():
    def gen():
        yield "$1"
        raise AssertionError("consumed past the first chunk")

    assert next(render_safe_stream(gen())) == "\\$1"
