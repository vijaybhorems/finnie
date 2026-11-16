"""Scoring for the speech-to-text eval: word error rate, key terms, numbers.

Both sides are normalised the same way first, so formatting differences
("401(k)" vs "401k", "$6,500" vs "6500", "S&P" vs "S and P") aren't errors.
"""
from __future__ import annotations

import re

_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def normalize(text: str) -> list[str]:
    t = text.lower()
    t = re.sub(r"\b(401|403|457)\s*\(?\s*([kb])\s*\)?(?![a-z])", r"\1\2", t)
    t = re.sub(r"\bs\s*(?:&|and)\s*p\b", "s and p", t)
    t = re.sub(r"\bp\s*/\s*e\b|\bp\.\s*e\.", "pe", t)
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 6,500 -> 6500
    t = t.replace("%", " percent ").replace("$", " ")
    t = re.sub(r"[^a-z0-9. ]+", " ", t)      # hyphens, apostrophes, punctuation
    t = re.sub(r"\.(?!\d)", " ", t)           # full stops, not decimal points
    return t.split()


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """(word edits, reference words): sum both over a set for the overall rate."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[len(hyp)], len(ref)


def has_term(term: str, text: str) -> bool:
    """The key term, as whole words at the start (so "etf" matches "etfs")."""
    words = " ".join(normalize(term))
    return re.search(rf"(?<![a-z0-9]){re.escape(words)}", " ".join(normalize(text))) is not None


def has_number(number: str, text: str) -> bool:
    tokens = normalize(text)
    if number in tokens:
        return True
    value = float(number)
    for scale, word in ((1_000, "thousand"), (1_000_000, "million")):
        scaled = value * scale
        if (str(int(scaled)) if scaled.is_integer() else str(scaled)) in tokens:  # 1.5 said back as 1500000
            return True
        for i, tok in enumerate(tokens[:-1]):  # 1500000 said back as "1.5 million"
            if tokens[i + 1] == word and _num(tok) is not None and _num(tok) * scale == value:
                return True
    return value.is_integer() and int(value) <= 90 and any(_SMALL.get(t) == int(value) for t in tokens)


def _num(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None
