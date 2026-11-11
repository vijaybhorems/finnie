"""Turn Finnie's written answer into something fit to speak.

Spike quality: enough to stop markdown being read aloud and to keep the full
written disclaimer off the audio (the session-start line covers it). The saved
answer is never changed; only what goes to text-to-speech.
"""
from __future__ import annotations

import re

# The session-start line from the design doc (Open question 3).
SESSION_DISCLAIMER = "I offer general financial education, not personal advice."

# What BaseAgent._add_disclaimer appends; spoken once per session instead.
_WRITTEN_DISCLAIMER = re.compile(
    r"\s*-{3,}\s*\*?Disclaimer:.*?(\*|$)", re.IGNORECASE | re.DOTALL
)
_MARKDOWN = [
    (re.compile(r"```.*?```", re.DOTALL), " "),          # code blocks: never speak
    (re.compile(r"`([^`]*)`"), r"\1"),                    # inline code
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),           # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),        # links -> their text
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),             # bold
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*"), r"\1"),  # italics
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),     # headings
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),          # bullets
    (re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE), ""),        # numbered lists
    (re.compile(r"^\s*>\s?", re.MULTILINE), ""),              # quotes
    (re.compile(r"^\s*\|?\s*:?-{3,}.*$", re.MULTILINE), ""),  # table rules
    (re.compile(r"\|"), ", "),                                # table cells
    (re.compile(r"^\s*-{3,}\s*$", re.MULTILINE), ""),         # horizontal rules
]


def strip_written_disclaimer(text: str) -> str:
    return _WRITTEN_DISCLAIMER.sub("", text)


def for_speech(text: str) -> str:
    """Remove markdown so the text reads naturally aloud."""
    for pattern, repl in _MARKDOWN:
        text = pattern.sub(repl, text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text)


class SpeechBuffer:
    """Streams tokens out in speakable pieces.

    Markdown markers can split across tokens (``*`` then ``*Roth**``), so text
    is held until a line or sentence ends, cleaned, then released. That costs a
    little time to first audio, measured in the spike like everything else.
    """

    _BOUNDARY = re.compile(r"(\n|[.!?:;](?=\s))")

    def __init__(self) -> None:
        self._pending = ""

    def push(self, token: str) -> str:
        self._pending += token
        match = None
        for match in self._BOUNDARY.finditer(self._pending):
            pass
        if match is None:
            return ""
        ready, self._pending = self._pending[: match.end()], self._pending[match.end():]
        return for_speech(ready)

    def flush(self) -> str:
        ready, self._pending = self._pending, ""
        return for_speech(strip_written_disclaimer(ready))
