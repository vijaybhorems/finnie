"""Who may join a voice call, checked by the worker when each call starts.

The access token is signed with our LiveKit secret, so its identity, room name
and metadata are what the app put there. They are still re-checked here against
the current settings: a token outlives the decision that minted it (10 minutes),
and a check in one place keeps a minting mistake from becoming a data leak.

  identity   "g-<Google sub>", the user id every piece of their data is filed under
  room       "<identity>:<32 hex>", which is also the call's conversation thread id
  metadata   {"email": "..."}, for the allowlist (the user id holds no email)

Pure: no LiveKit or database imports, so the app and CI can use it.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from src.core.config import VoiceConfig, get_settings

# Spoken, then the call ends, when a join is refused. Says nothing about why.
VOICE_UNAVAILABLE = "Voice isn't available right now."

_IDENTITY = re.compile(r"g-[A-Za-z0-9_-]{1,255}")
_ROOM_SUFFIX = re.compile(r"[0-9a-f]{32}")


def new_voice_room(user_id: str) -> str:
    """A fresh room name for one call by `user_id`; also the call's thread id."""
    if not _IDENTITY.fullmatch(user_id or ""):
        raise ValueError("voice needs a Google user id (g-<sub>)")
    return f"{user_id}:{uuid.uuid4().hex}"


def token_metadata(email: str) -> str:
    """The participant metadata the app signs into a voice token."""
    return json.dumps({"email": (email or "").strip().lower()})


def _email_from(metadata: str) -> str:
    try:
        data = json.loads(metadata or "{}")
    except (TypeError, ValueError):
        return ""
    email = data.get("email") if isinstance(data, dict) else None
    return email.strip().lower() if isinstance(email, str) else ""


def is_allowed(email: str, voice: Optional[VoiceConfig] = None) -> bool:
    """Whether `email` may use voice now. An empty allowlist allows nobody."""
    voice = voice or get_settings().voice
    if not voice.enabled or not email:
        return False
    allowed = voice.allowed_emails
    return "*" in allowed or email.strip().lower() in allowed


def check_join(identity: str, room: str, metadata: str, voice: Optional[VoiceConfig] = None) -> Optional[str]:
    """None if this participant may hold this call, else the reason it may not.

    The reason is for logs only; the caller hears VOICE_UNAVAILABLE whatever it is.
    """
    voice = voice or get_settings().voice
    if not voice.enabled:
        return "voice_disabled"
    if not _IDENTITY.fullmatch(identity or ""):
        return "bad_identity"
    owner, sep, suffix = (room or "").rpartition(":")
    if not sep or owner != identity or not _ROOM_SUFFIX.fullmatch(suffix):
        return "room_not_owned"
    if not is_allowed(_email_from(metadata), voice):
        return "not_allowlisted"
    return None
