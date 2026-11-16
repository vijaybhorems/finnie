"""Speech-to-text settings: one place, so the worker and the eval use the same.

Pure (no LiveKit). The worker passes `deepgram_options()` to LiveKit's Deepgram
plugin; tests/evals/test_stt_evals.py sends the same options to Deepgram's
prerecorded API, so a change here is what the eval measures.

`clean_transcript()` turns finance terms that come back spelled out ("four
zero one k") into the written form the agents and the classifier expect.
"""
from __future__ import annotations

import re
from typing import Any

# Words Deepgram should favour: finance terms it otherwise mishears ("Roth" as
# "wrath", "RSI" as "or a sigh"). Re-run the speech-to-text eval after changing.
KEYTERMS = [
    "Roth", "Roth IRA", "IRA", "401(k)", "403(b)", "HSA", "ETF", "RSI", "MACD",
    "P/E ratio", "S&P 500", "Nasdaq", "Dow Jones", "dividend", "capital gains",
    "tax-loss harvesting", "wash sale", "FIRE", "Finnie",
    # Added after the first speech-to-text eval: heard as "reeds", "Eyebombs",
    # "CD letter", "Bakshire".
    "REIT", "I bonds", "CD ladder", "Berkshire Hathaway", "529 plan",
]

# Even with key terms, Deepgram writes some account names as the words said
# (tests/evals/test_stt_evals.py): "four zero one k", "five twenty nine".
_D = r"(?:zero|oh|o|0)"
_SPOKEN_TERMS = [
    (re.compile(rf"\b(?:four|4)[\s-]+{_D}[\s-]+(?:one|1)[\s-]*k'?s?\b", re.IGNORECASE), "401(k)"),
    (re.compile(rf"\b(?:four|4)[\s-]+{_D}[\s-]+(?:three|3)[\s-]*b'?s?\b", re.IGNORECASE), "403(b)"),
    (re.compile(r"\b(?:four|4)[\s-]+(?:fifty|50)[\s-]+(?:seven|7)[\s-]*b\b", re.IGNORECASE), "457(b)"),
    (re.compile(r"\b(?:five|5)[\s-]+(?:(?:twenty|20)[\s-]+(?:nine|9)|29)\b", re.IGNORECASE), "529"),
    (re.compile(r"\bS\s+and\s+P(?:\s+(?:five\s+hundred|500))?\b", re.IGNORECASE),
     lambda m: "S&P 500" if len(m.group(0).split()) > 3 else "S&P"),
]


def clean_transcript(text: str) -> str:
    """The transcript with spelled-out finance terms written the usual way."""
    for pattern, repl in _SPOKEN_TERMS:
        text = pattern.sub(repl, text)
    return text



def deepgram_options(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "language": "en-US",
        "keyterm": list(KEYTERMS),
        "smart_format": True,  # numbers as digits, so is_sensitive() can see them
        "mip_opt_out": True,   # Deepgram's model-improvement opt-out (design doc, Impact item 8)
    }
