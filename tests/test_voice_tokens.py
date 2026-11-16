"""src/voice/tokens.py: who gets a call token, how often, and what it allows."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from livekit import api

from src.core.config import get_settings
from src.voice import tokens
from src.voice.tokens import VoiceRefused, mint_call_token

USER = "g-1234567890"
EMAIL = "alice@example.com"
KEY, SECRET = "testkey", "a-test-secret-that-is-long-enough-for-hs256"
NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def voice_on(monkeypatch):
    """Voice enabled for alice, LiveKit configured, a fresh in-memory store, no real rooms."""
    from src.persistence.backend import reset_persistence

    monkeypatch.setenv("FINNIE_VOICE_ENABLED", "1")
    monkeypatch.setenv("FINNIE_VOICE_ALLOWED_EMAILS", EMAIL)
    monkeypatch.setenv("LIVEKIT_URL", "wss://lk.example.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", KEY)
    monkeypatch.setenv("LIVEKIT_API_SECRET", SECRET)
    get_settings.cache_clear()
    reset_persistence()
    closer = AsyncMock(return_value=0)
    monkeypatch.setattr(tokens, "_close_other_calls", closer)
    yield closer
    reset_persistence()


def _claims(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms=["HS256"], options={"verify_nbf": False, "verify_exp": False})


def _refusal(**kw) -> str:
    with pytest.raises(VoiceRefused) as info:
        mint_call_token(kw.pop("user", USER), kw.pop("email", EMAIL), now=kw.pop("now", NOW))
    return info.value.reason


class TestWhoGetsAToken:
    def test_allowlisted_user_gets_one(self):
        call = mint_call_token(USER, EMAIL, now=NOW)
        assert call.url == "wss://lk.example.test"
        assert call.room.startswith(f"{USER}:")

    def test_not_on_the_allowlist(self):
        assert _refusal(email="bob@example.com") == "not_allowed"

    def test_voice_switched_off(self, monkeypatch):
        monkeypatch.setenv("FINNIE_VOICE_ENABLED", "0")
        get_settings.cache_clear()
        assert _refusal() == "not_allowed"

    def test_not_a_google_user_id(self):
        assert _refusal(user="e-abc") == "not_allowed"

    def test_livekit_not_configured(self, monkeypatch):
        monkeypatch.delenv("LIVEKIT_API_SECRET")
        assert _refusal() == "not_configured"

    def test_refusal_messages_hide_how_access_is_decided(self):
        assert VoiceRefused("not_allowed").message == "Voice isn't available for your account."
        assert "allowlist" not in " ".join(tokens.REFUSALS.values()).lower()


class TestLimits:
    def test_ten_tokens_an_hour(self):
        for i in range(10):
            mint_call_token(USER, EMAIL, now=NOW + timedelta(minutes=i))
        assert _refusal(now=NOW + timedelta(minutes=30)) == "rate_limited"
        # The first expires an hour after it was minted.
        assert mint_call_token(USER, EMAIL, now=NOW + timedelta(minutes=60, seconds=1))

    def test_rate_limit_is_per_user(self):
        for i in range(10):
            mint_call_token(USER, EMAIL, now=NOW + timedelta(minutes=i))
        assert mint_call_token("g-999", EMAIL, now=NOW + timedelta(minutes=11))

    def test_no_token_once_todays_minutes_are_used(self):
        from src.persistence.user_data import UserData

        UserData(USER).add_voice_seconds(30 * 60, now=NOW)
        assert _refusal() == "daily_limit"
        assert mint_call_token(USER, EMAIL, now=NOW + timedelta(days=1))  # a new UTC day

    def test_refused_requests_do_not_use_up_the_hour(self):
        from src.persistence.user_data import UserData

        _refusal(email="bob@example.com")
        assert UserData(USER).voice_tokens_last_hour(NOW) == 0


class TestWhatTheTokenAllows:
    def test_one_room_microphone_only_for_ten_minutes(self):
        call = mint_call_token(USER, EMAIL, now=NOW)
        claims = _claims(call.token)
        video = claims["video"]
        assert claims["sub"] == USER and claims["iss"] == KEY
        assert video["roomJoin"] is True and video["room"] == call.room
        assert video["canPublishSources"] == ["microphone"]
        assert video["canPublishData"] is False
        for admin in ("roomCreate", "roomList", "roomAdmin", "roomRecord", "ingressAdmin"):
            assert not video.get(admin)
        assert claims["exp"] - claims["nbf"] == 10 * 60
        assert json.loads(claims["metadata"]) == {"email": EMAIL}

    def test_livekit_verifies_it(self):
        call = mint_call_token(USER, EMAIL, now=NOW)
        verified = api.TokenVerifier(KEY, SECRET).verify(call.token)
        assert verified.identity == USER and verified.video.room == call.room

    def test_every_call_gets_a_new_room(self):
        assert mint_call_token(USER, EMAIL, now=NOW).room != mint_call_token(USER, EMAIL, now=NOW).room


class TestOneCallPerUser:
    def test_other_rooms_of_this_user_are_closed(self, voice_on):
        call = mint_call_token(USER, EMAIL, now=NOW)
        voice_on.assert_awaited_once()
        args = voice_on.await_args.args
        assert args[3] == USER and args[4] == call.room

    def test_a_failed_close_does_not_block_the_call(self, voice_on):
        voice_on.side_effect = OSError("livekit down")
        assert mint_call_token(USER, EMAIL, now=NOW).room

    async def test_minting_inside_a_running_loop_still_closes(self, voice_on):
        # The async scripted caller mints from inside its event loop.
        voice_on.return_value = 2
        assert tokens.close_other_calls(USER, keep="x") == 2

    async def test_close_deletes_only_this_users_other_rooms(self, monkeypatch):
        monkeypatch.undo()  # the real _close_other_calls, with a fake LiveKit API
        lk = MagicMock()
        rooms = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        for room, name in zip(rooms, [f"{USER}:{'a' * 32}", f"{USER}:{'b' * 32}", f"g-999:{'c' * 32}", f"{USER}9:{'d' * 32}"]):
            room.name = name
        lk.room.list_rooms = AsyncMock(return_value=MagicMock(rooms=rooms))
        lk.room.delete_room = AsyncMock()
        lk.aclose = AsyncMock()
        monkeypatch.setattr(tokens.api, "LiveKitAPI", lambda *a: lk)

        closed = await tokens._close_other_calls("wss://x", KEY, SECRET, USER, keep=f"{USER}:{'b' * 32}")
        assert closed == 1
        assert [c.args[0].room for c in lk.room.delete_room.await_args_list] == [f"{USER}:{'a' * 32}"]
        lk.aclose.assert_awaited_once()
