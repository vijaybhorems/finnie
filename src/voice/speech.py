"""Turn Finnie's written answer into something fit to speak.

Enough to stop markdown being read aloud and to keep the full
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


# Discreet mode (design doc, Open question 4): dollar figures are not spoken.
# The prompt asks for percentages instead; this catches any that get through,
# as is_sensitive() backs up the memory prompt. The full answer, figures
# included, is saved and shown on screen.
AMOUNT_ON_SCREEN = "the amount on your screen"

_NUM = r"\d[\d,]*(?:\.\d+)?"
_SCALE = r"(?:thousand|million|billion|trillion)"
_WORD_NUM = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|trillion)"
)
_AMOUNTS = [
    # $1,234.56  $12k  $1.2M  $3 million  £500  US$40
    re.compile(rf"(?:US)?[$£€]\s?{_NUM}(?:\s?(?:k|K|m|M|mm|bn|B)\b|\s{_SCALE}\b)?"),
    # USD 500, 500 USD, 2,000 dollars, 1.5 million dollars, 20 bucks
    re.compile(rf"\bUSD\s?{_NUM}(?:\s{_SCALE}\b)?"),
    re.compile(rf"\b{_NUM}(?:\s{_SCALE})?\s(?:dollars?|bucks|USD)\b", re.IGNORECASE),
    # two thousand dollars, a few hundred dollars, fifty-five dollars
    re.compile(
        rf"\b(?:(?:{_WORD_NUM}|a|and|few|several|couple|of)[\s-]+)*{_WORD_NUM}[\s-]+(?:dollars?|bucks)\b",
        re.IGNORECASE,
    ),
]
# "the amount on your screen to the amount on your screen" -> said once
_RANGE = re.compile(rf"{AMOUNT_ON_SCREEN}(?:\s*(?:to|and|or|-|–)\s*{AMOUNT_ON_SCREEN})+")
# "about the amount on your screen" -> no rounding words before a hidden figure
_HEDGE = re.compile(rf"\b(?:about|roughly|approximately)\s+(?={AMOUNT_ON_SCREEN})", re.IGNORECASE)


def mask_amounts(text: str) -> str:
    """Replace dollar figures with AMOUNT_ON_SCREEN, for discreet speech."""
    for pattern in _AMOUNTS:
        text = pattern.sub(AMOUNT_ON_SCREEN, text)
    return _HEDGE.sub("", _RANGE.sub(AMOUNT_ON_SCREEN, text))


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
