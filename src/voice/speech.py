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


# Said the way a listener expects (design doc, Evaluation layer 1). ElevenLabs
# reads most text well; these are the forms it gets wrong or reads oddly.
TICKER_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "GOOG": "Alphabet", "META": "Meta", "TSLA": "Tesla",
    "BRK.B": "Berkshire Hathaway", "BRK.A": "Berkshire Hathaway", "JPM": "JPMorgan",
    "V": "Visa", "MA": "Mastercard", "JNJ": "Johnson and Johnson", "WMT": "Walmart",
    "XOM": "Exxon", "UNH": "UnitedHealth", "HD": "Home Depot", "PG": "Procter and Gamble",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "COST": "Costco", "DIS": "Disney", "NFLX": "Netflix",
    "AMD": "AMD", "INTC": "Intel", "BAC": "Bank of America", "AVGO": "Broadcom",
    # Funds: their letters, which is how people say them.
    "SPY": "S P Y", "QQQ": "triple Q", "VOO": "V O O", "VTI": "V T I", "DIA": "D I A",
    "IWM": "I W M", "BND": "B N D", "AGG": "A G G",
}
# Single letters that are tickers only in a ticker-shaped spot: "(V)", "V shares".
_SHORT_TICKERS = {t for t in TICKER_NAMES if len(t) <= 2}
_ACCOUNTS = [
    (re.compile(r"\b401\s?\(?k\)?(?=\W|$)", re.IGNORECASE), "four-oh-one-k"),
    (re.compile(r"\b403\s?\(?b\)?(?=\W|$)", re.IGNORECASE), "four-oh-three-b"),
    (re.compile(r"\b457\s?\(?b\)?(?=\W|$)", re.IGNORECASE), "four-fifty-seven-b"),
    (re.compile(r"\b529(?=\s+(?:plans?|accounts?|savings))", re.IGNORECASE), "five-twenty-nine"),
]
_SUFFIX_AMOUNT = re.compile(rf"(?:US)?\$\s?({_NUM})\s?(k|K|m|M|mm|bn|B|b)\b")
_SUFFIX_WORDS = {"k": "thousand", "m": "million", "mm": "million", "b": "billion", "bn": "billion"}
_SYMBOLS = [
    (re.compile(r"\bS&P\b"), "S and P"),
    (re.compile(r"\bP/E\b"), "P E"),
    (re.compile(r"\bvs\.?(?=\s)", re.IGNORECASE), "versus"),
    (re.compile(r"\be\.g\.,?", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.,?", re.IGNORECASE), "that is"),
    (re.compile(r"\bYoY\b"), "year over year"),
    (re.compile(r"\bYTD\b"), "year to date"),
    (re.compile(r"(?<=\d)\s?bps\b"), " basis points"),
    (re.compile(r"~\s?(?=\d|\$)"), "about "),
]
# "Apple (AAPL)" -> "Apple": the name is already said.
_NAME_THEN_TICKER = re.compile(r"\s*\((?:NASDAQ:|NYSE:)?\s*\$?([A-Z]{1,5}(?:\.[AB])?)\)")
_TICKER = re.compile(r"(?<![\w$.])\$?([A-Z]{1,5}(?:\.[AB])?)(?!\w|\.\w)")


def _ticker_name(match: re.Match) -> str:
    symbol = match.group(1)
    name = TICKER_NAMES.get(symbol)
    if name is None or (symbol in _SHORT_TICKERS and not match.group(0).startswith("$")):
        return match.group(0)
    return name


def normalize_for_speech(text: str) -> str:
    """Rewrite forms text-to-speech reads badly: "$1.2M", "401(k)", AAPL, "S&P".

    Runs after mask_amounts, so in discreet mode amounts are already gone.
    One- and two-letter tickers (V, MA, KO) are only renamed with a "$" in
    front, since bare they are usually ordinary words or initials.
    """
    text = _NAME_THEN_TICKER.sub(
        lambda m: "" if m.group(1) in TICKER_NAMES else m.group(0), text,
    )
    text = _SUFFIX_AMOUNT.sub(
        lambda m: f"{m.group(1)} {_SUFFIX_WORDS[m.group(2).lower()]} dollars", text,
    )
    for pattern, repl in (*_ACCOUNTS, *_SYMBOLS):
        text = pattern.sub(repl, text)
    return _TICKER.sub(_ticker_name, text)


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
    # The written disclaimer always ends the answer. When it arrives in
    # pieces rather than as one chunk, stop at its first line: the rest is
    # never spoken (the session-start line covers it).
    _DISCLAIMER_START = re.compile(r"(?:^|\n)\s*(?:-{3,}\s*)?\*?\s*Disclaimer:", re.IGNORECASE)

    def __init__(self) -> None:
        self._pending = ""
        self._done = False  # reached the written disclaimer

    def _release(self, ready: str) -> str:
        start = self._DISCLAIMER_START.search(ready)
        if start is not None:
            ready, self._done, self._pending = ready[: start.start()], True, ""
        return for_speech(ready)

    def push(self, token: str) -> str:
        if self._done:
            return ""
        self._pending += token
        match = None
        for match in self._BOUNDARY.finditer(self._pending):
            pass
        if match is None:
            return ""
        ready, self._pending = self._pending[: match.end()], self._pending[match.end():]
        return self._release(ready)

    def flush(self) -> str:
        if self._done:
            return ""
        ready, self._pending = self._pending, ""
        return self._release(strip_written_disclaimer(ready))
