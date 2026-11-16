"""Starting a voice call: the LiveKit token the app hands a signed-in user.

Called server-side by the page, with the identity from the sign-in session
(src/web_app/session.py), never from browser input. The design doc's rules:

- only a user with voice enabled and on the allowlist, and with voice minutes
  left today, gets a token, at most `voice.tokens_per_hour` per hour;
- one new room per call, named <user_id>:<32 hex>, which is also the call's
  thread id;
- the token joins that one room and publishes microphone audio only: no
  video, screen share or data messages, no creating or listing rooms;
- it must be used within `voice.token_ttl_minutes`; a joined call continues;
- one call per user: any other open room of this user's is closed first.

The worker checks all of this again when the call joins (access.py).
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Coroutine, Optional

from livekit import api

from src.core.config import get_settings
from src.utils.logger import get_logger
from src.voice.access import is_allowed, new_voice_room, token_metadata
from src.voice.limits import DAILY_LIMIT

logger = get_logger(__name__)

# What the page shows for each refusal. Only `rate_limited` and `daily_limit`
# say why: the others would tell an outsider how access is decided.
REFUSALS = {
    "not_allowed": "Voice isn't available for your account.",
    "not_configured": "Voice isn't available right now.",
    "daily_limit": DAILY_LIMIT,
    "rate_limited": "You've started a lot of calls in the last hour. Try again a little later.",
}


class VoiceRefused(Exception):
    """No token: `reason` for logs and tests, `message` for the page."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.message = REFUSALS[reason]


@dataclass(frozen=True)
class CallToken:
    url: str     # the LiveKit server the browser connects to
    token: str   # signed JWT, valid to join for voice.token_ttl_minutes
    room: str    # <user_id>:<32 hex>, also the call's thread id


def _livekit_credentials() -> tuple[str, str, str]:
    """URL, key and secret from the environment; no dev defaults in the app."""
    url = os.environ.get("LIVEKIT_URL", "").strip()
    key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    if not (url and key and secret):
        raise VoiceRefused("not_configured")
    return url, key, secret


def build_token(identity: str, room: str, email: str, key: str, secret: str, ttl: timedelta) -> str:
    """A token that joins `room` as `identity` and publishes microphone audio only."""
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=False,
        can_publish_sources=["microphone"],
        # room_create, room_list, room_admin and recording stay off (the defaults).
    )
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_metadata(token_metadata(email))
        .with_grants(grants)
        .with_ttl(ttl)
        .to_jwt()
    )


async def _close_other_calls(url: str, key: str, secret: str, user_id: str, keep: str) -> int:
    """Delete every open room of `user_id` except `keep`; returns how many."""
    prefix = f"{user_id}:"
    lk = api.LiveKitAPI(url, key, secret)
    try:
        rooms = (await lk.room.list_rooms(api.ListRoomsRequest())).rooms
        stale = [r.name for r in rooms if r.name.startswith(prefix) and r.name != keep]
        for name in stale:
            await lk.room.delete_room(api.DeleteRoomRequest(room=name))
        return len(stale)
    finally:
        await lk.aclose()


def _run(coro: Coroutine[Any, Any, int]) -> int:
    """Run `coro` to completion from sync code, even if this thread has a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def close_other_calls(user_id: str, keep: str = "") -> int:
    """Synchronous wrapper for the page. Failures are logged; the new call goes ahead."""
    try:
        url, key, secret = _livekit_credentials()
        return _run(_close_other_calls(url, key, secret, user_id, keep))
    except VoiceRefused:
        return 0
    except Exception as exc:  # noqa: BLE001 — a stale room ends at its own limits
        logger.warning("voice_close_other_calls_failed", error=type(exc).__name__)
        return 0


def mint_call_token(user_id: str, email: str, *, now: Optional[datetime] = None) -> CallToken:
    """A token for one new call by the signed-in user, or VoiceRefused."""
    from src.persistence.user_data import UserData

    voice = get_settings().voice
    if not is_allowed(email, voice):
        raise VoiceRefused("not_allowed")
    try:
        room = new_voice_room(user_id)
    except ValueError:
        raise VoiceRefused("not_allowed") from None
    url, key, secret = _livekit_credentials()

    data = UserData(user_id)
    if data.voice_seconds_today(now) >= voice.daily_minutes_per_user * 60:
        raise VoiceRefused("daily_limit")
    if data.voice_tokens_last_hour(now) >= voice.tokens_per_hour:
        logger.warning("voice_token_rate_limited")
        raise VoiceRefused("rate_limited")
    data.record_voice_token(now)

    closed = close_other_calls(user_id, keep=room)
    token = build_token(user_id, room, email, key, secret, timedelta(minutes=voice.token_ttl_minutes))
    logger.info("voice_token_minted", closed_other_calls=closed)
    return CallToken(url=url, token=token, room=room)
