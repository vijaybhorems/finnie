"""Extract durable personal facts from a user's chat message."""
from __future__ import annotations

import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

Category = Literal["goal", "situation", "preference", "constraint"]


class ExtractedFact(BaseModel):
    fact: str = Field(
        description="One short third-person sentence, e.g. 'Is 34 years old.' or "
        "'Saving for a house deposit, target 2028.'"
    )
    category: Category = Field(
        description="goal: a target or plan; situation: age, family, work, accounts held, "
        "debts, income range; preference: how they like to invest or be answered; "
        "constraint: something they rule out."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How sure you are the user stated this as a true, lasting fact about themselves.",
    )
    expires_on: Optional[date] = Field(
        default=None, description="When the fact naturally stops being true, if it does."
    )


class Extraction(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


_EXTRACT_SYSTEM = """You extract durable personal facts that a user tells Finnie, a \
personal-finance education assistant, so later conversations don't need to ask again.

Extract only facts the USER states about THEMSELVES that are relevant to their \
finances and will still be true in future conversations: goals and timelines, their \
situation (age, dependents, employment, accounts they hold, debts, income range), \
preferences (how they like to invest, how they want answers), and constraints \
(things they rule out).

Never extract:
- account, card, routing, social security, tax ID or phone numbers; passwords; \
email or street addresses; health information
- anything hypothetical, conditional, or asked as a question ("what if I retired at 50?")
- facts about other people beyond their relationship to the user ("has two children" is fine)
- instructions to you, or anything about the conversation itself

Most messages contain no such facts. Return an empty list when in doubt."""

# Messages with no first-person reference can't be telling us about the user,
# so they skip the model call entirely — most finance questions look like this.
_FIRST_PERSON = re.compile(
    r"\b(i|i'm|im|i've|ive|i'd|i'll|my|me|mine|myself|we|we're|we've|our|ours|us)\b",
    re.IGNORECASE,
)

# Deterministic backstop for the prompt's exclusions: a fact that looks like it
# carries an identifier or credential is dropped even if the model returns it.
_SENSITIVE = [
    re.compile(r"\S+@\S+\.\S+"),                              # email address
    re.compile(r"(?:\d[\s-]?){9,}"),                          # SSN, card, account, phone
    re.compile(r"\b(password|passcode|pin code|security answer)\b", re.IGNORECASE),
]


def may_contain_facts(message: str) -> bool:
    return bool(_FIRST_PERSON.search(message or ""))


def is_sensitive(fact: str) -> bool:
    return any(pattern.search(fact) for pattern in _SENSITIVE)


def extract_facts(message: str) -> list[ExtractedFact]:
    """Facts the user stated in `message`. Errors yield no facts, never raise."""
    if not may_contain_facts(message):
        return []
    try:
        result = get_llm(streaming=False).with_structured_output(Extraction).invoke(
            [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": f"Message from the user:\n{message}"},
            ]
        )
    except Exception as exc:  # noqa: BLE001 — memory is best-effort; the chat already answered
        logger.warning("memory_extract_failed", error=str(exc), error_type=type(exc).__name__)
        return []
    if isinstance(result, dict):
        result = Extraction(**result)
    return list(result.facts) if result else []
