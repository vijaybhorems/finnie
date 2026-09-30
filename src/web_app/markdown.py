"""Make model output safe to pass to Streamlit markdown.

Streamlit treats text between two ``$`` as inline LaTeX, so "trades at $40 ...
earned $2" renders the span between them as math. Escape at render time only:
stored answers (chat history, FAQ cache, evals) keep the model's raw text.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator


def render_safe(text: str) -> str:
    """Escape ``$`` so Streamlit markdown shows it literally instead of as LaTeX."""
    return text.replace("$", r"\$")


def render_safe_stream(chunks: Iterable[str]) -> Iterator[str]:
    """Apply `render_safe` to each chunk of a stream fed to ``st.write_stream``.

    Per-chunk escaping is enough because it is per character: a ``$`` is
    escaped the same whichever chunk it lands in.
    """
    for chunk in chunks:
        yield render_safe(chunk)
