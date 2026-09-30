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
  settings  {memory_enabled}                         (user's memory switch)

Memories live one level down, in ("users", <user_id>, "memories"), one item
per fact, semantically indexed on "fact" so the relevant few can be recalled
per question. Only memory items are indexed; everything above is written with
index=False and never pays for an embedding.

Values are sanitised on write: Postgres jsonb rejects NaN, which is exactly
what st.data_editor returns for a half-filled row.
"""
from __future__ import annotations

import math
import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional

from langgraph.store.base import BaseStore

from src.core.config import get_settings
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
        self._store.put(self._namespace, key, _json_safe(value), index=False)

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

    # ── Memories ─────────────────────────────────────────────────────────────

    @property
    def _memory_namespace(self) -> tuple[str, str, str]:
        return ("users", self._user_id, "memories")

    def memory_enabled(self) -> bool:
        """The user's own switch; on unless they turned it off."""
        return bool((self._get("settings") or {}).get("memory_enabled", True))

    def set_memory_enabled(self, enabled: bool) -> None:
        self._put("settings", {**(self._get("settings") or {}), "memory_enabled": bool(enabled)})

    @staticmethod
    def _is_expired(value: dict[str, Any]) -> bool:
        expires_on = value.get("expires_on")
        return bool(expires_on) and str(expires_on) < date.today().isoformat()

    def _live(self, items: list[Any]) -> list[dict[str, Any]]:
        """Items as memory dicts, deleting any that have expired."""
        live = []
        for item in items:
            if self._is_expired(item.value):
                self._store.delete(self._memory_namespace, item.key)
                continue
            memory = {"id": item.key, **item.value}
            if getattr(item, "score", None) is not None:
                memory["relevance"] = float(item.score)
            live.append(memory)
        return live

    def list_memories(self) -> list[dict[str, Any]]:
        """Every live memory, newest first — what the user sees in the panel."""
        cap = get_settings().memory.max_memories_per_user
        items = self._store.search(self._memory_namespace, limit=cap * 2)
        return sorted(self._live(items), key=lambda m: m.get("created_at", ""), reverse=True)

    def relevant_memories(self, query: str, k: Optional[int] = None) -> list[dict[str, Any]]:
        """The k memories most relevant to `query` (empty if memory is off)."""
        if not query.strip() or not self.memory_enabled():
            return []
        k = k or get_settings().memory.top_k
        items = self._store.search(self._memory_namespace, query=query, limit=k * 2)
        return self._live(items)[:k]

    def remember(
        self,
        fact: str,
        category: str,
        confidence: float,
        expires_on: Optional[str] = None,
    ) -> str:
        """Save a fact, replacing a near-identical one of the same category.

        Returns the memory id. Replacement keeps one current version of a fact
        ("Is 35" supersedes "Is 34") instead of contradictory copies.
        """
        settings = get_settings().memory
        similar = self._store.search(
            self._memory_namespace, query=fact, filter={"category": category}, limit=1
        )
        replaces = similar and (similar[0].score or 0.0) >= settings.dedup_similarity
        memory_id = similar[0].key if replaces else uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        self._store.put(
            self._memory_namespace,
            memory_id,
            _json_safe({
                "fact": fact,
                "category": category,
                "confidence": float(confidence),
                "noted_on": now.date().isoformat(),
                "created_at": now.isoformat(),
                "expires_on": expires_on,
            }),
        )
        self._enforce_memory_cap()
        return memory_id

    def _enforce_memory_cap(self) -> None:
        cap = get_settings().memory.max_memories_per_user
        memories = self.list_memories()
        for stale in memories[cap:]:
            self._store.delete(self._memory_namespace, stale["id"])

    def forget(self, memory_id: str) -> None:
        """Delete one memory — a real delete, not a flag."""
        self._store.delete(self._memory_namespace, memory_id)

    def forget_all(self) -> int:
        memories = self.list_memories()
        for memory in memories:
            self._store.delete(self._memory_namespace, memory["id"])
        return len(memories)

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
