"""Stable user identity from the authenticated Google account."""
from __future__ import annotations

import hashlib
from typing import Optional


def user_id_from_claims(sub: Optional[str], email: Optional[str]) -> str:
    """Return the id every piece of a user's data is filed under.

    Prefers Google's `sub` claim: stable for the life of the account and, unlike
    the email address, never reassigned and not personal data in itself. Falls
    back to a hash of the email only if a provider omits `sub`.

    The result is used as a LangGraph store namespace label, which may not
    contain periods, so neither form includes the raw email.
    """
    if sub and str(sub).strip():
        return f"g-{str(sub).strip()}"
    if email and email.strip():
        digest = hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:32]
        return f"e-{digest}"
    raise ValueError("cannot derive a user id: identity has neither 'sub' nor 'email'")
