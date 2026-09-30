"""Everything Finnie stores for one user.

UserData is bound to a single user id at construction and exposes no method
that accepts another, so code holding one cannot reach a different user's data
by accident. Construct it only from the server-side authenticated identity
(src/persistence/identity.py), never from a value the browser supplies.

Layout in the LangGraph store, namespace ("users", <user_id>):

  profile   {knowledge_level, risk_tolerance, investment_horizon}
  holdings  {"items": [{ticker, shares, avg_cost}, ...]}
  plan      {start_age, horizon, current_savings, monthly_contribution,
             risk_tolerance, events: [...]}          (Goals → Life Timeline)
  session   {active_thread_id}                       (current chat thread)

Values are sanitised on write: Postgres jsonb rejects NaN, which is exactly
what st.data_editor returns for a half-filled row.
"""
from __future__ import annotations

import math
import uuid
from typing import Any, Optional

from langgraph.store.base import BaseStore

from src.core.state import UserProfile
from src.persistence.backend import get_store

KNOWLEDGE_LEVELS = ("beginner", "intermediate", "advanced")
RISK_TOLERANCES = ("conservative", "moderate", "aggressive")
INVESTMENT_HORIZONS = ("short", "medium", "long")

DEFAULT_PROFILE = {
    "knowledge_level": "beginner",
    "risk_tolerance": "moderate",
    "investment_horizon": "long",
}
_PROFILE_CHOICES = {
    "knowledge_level": KNOWLEDGE_LEVELS,
    "risk_tolerance": RISK_TOLERANCES,
    "investment_horizon": INVESTMENT_HORIZONS,
}


def _finite(value: Any) -> Optional[float]:
    """float(value) if it is a finite number, else None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    """Copy of `value` with non-finite floats replaced by 0.0 (jsonb rejects NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def clean_holdings(rows: Any) -> list[dict[str, Any]]:
    """Keep rows with a ticker and a finite share count; normalise their fields."""
    cleaned: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        shares = _finite(row.get("shares"))
        if not ticker or ticker == "NAN" or len(ticker) > 10 or shares is None:
            continue
        cleaned.append({
            "ticker": ticker,
            "shares": shares,
            "avg_cost": _finite(row.get("avg_cost")) or 0.0,
        })
    return cleaned


class UserData:
    """One user's saved profile, holdings, plan and chat thread."""

    def __init__(self, user_id: str, store: Optional[BaseStore] = None) -> None:
        if not user_id or "." in user_id:
            raise ValueError(f"invalid user id: {user_id!r}")
        self._user_id = user_id
        self._store = store if store is not None else get_store()

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def _namespace(self) -> tuple[str, str]:
        return ("users", self._user_id)

    def _get(self, key: str) -> Optional[dict[str, Any]]:
        item = self._store.get(self._namespace, key)
        return item.value if item else None

    def _put(self, key: str, value: dict[str, Any]) -> None:
        self._store.put(self._namespace, key, _json_safe(value))

    # ── Profile ──────────────────────────────────────────────────────────────

    def get_profile(self) -> dict[str, str]:
        """Saved profile, with defaults for anything missing or invalid."""
        saved = self._get("profile") or {}
        return {
            field: saved.get(field) if saved.get(field) in choices else DEFAULT_PROFILE[field]
            for field, choices in _PROFILE_CHOICES.items()
        }

    def save_profile(self, profile: dict[str, Any]) -> dict[str, str]:
        merged = self.get_profile()
        for field, choices in _PROFILE_CHOICES.items():
            if profile.get(field) in choices:
                merged[field] = profile[field]
        self._put("profile", merged)
        return merged

    # ── Holdings ─────────────────────────────────────────────────────────────

    def get_holdings(self) -> list[dict[str, Any]]:
        return clean_holdings((self._get("holdings") or {}).get("items"))

    def save_holdings(self, rows: Any) -> list[dict[str, Any]]:
        holdings = clean_holdings(rows)
        self._put("holdings", {"items": holdings})
        return holdings

    # ── Saved plan (Goals → Life Timeline) ───────────────────────────────────

    def get_plan(self) -> Optional[dict[str, Any]]:
        return self._get("plan")

    def save_plan(self, plan: dict[str, Any]) -> None:
        self._put("plan", plan)

    # ── Chat thread ──────────────────────────────────────────────────────────

    def _thread_prefix(self) -> str:
        return f"{self._user_id}:"

    def owns_thread(self, thread_id: str) -> bool:
        return bool(thread_id) and thread_id.startswith(self._thread_prefix())

    def active_thread_id(self) -> str:
        """The user's current chat thread, created on first use.

        Stored rather than kept in the browser session, so a page refresh or a
        second device lands back in the same conversation.
        """
        thread_id = (self._get("session") or {}).get("active_thread_id", "")
        return thread_id if self.owns_thread(thread_id) else self.new_thread()

    def new_thread(self) -> str:
        """Start a fresh conversation (the old one stays in the checkpointer)."""
        thread_id = f"{self._thread_prefix()}{uuid.uuid4().hex}"
        self._put("session", {"active_thread_id": thread_id})
        return thread_id

    # ── For the workflow ─────────────────────────────────────────────────────

    def to_user_profile(self) -> UserProfile:
        """The UserProfile agents see: saved profile, holdings and plan."""
        plan = self.get_plan()
        return UserProfile(
            user_id=self._user_id,
            **self.get_profile(),
            portfolio=self.get_holdings(),
            goals=[{"type": "life_timeline", **plan}] if plan else [],
        )
