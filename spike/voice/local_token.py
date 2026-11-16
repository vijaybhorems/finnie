"""Tokens for the local test tools (talk page, scripted caller). LOCAL USE ONLY.

`new_call()` goes through the app's own minter (src/voice/tokens.py: rate
limit, daily minutes, one call per user) as a made-up Google user g-local-<name>
with VOICE_TEST_EMAIL. `raw_call()` signs a token directly, skipping the
minter, to check that the worker refuses what the minter would not have issued.
Start the worker with voice on and that email allowed:
    FINNIE_VOICE_ENABLED=1 FINNIE_VOICE_ALLOWED_EMAILS=tester@example.com \
        .venv-voice/bin/python -m src.voice.worker dev
"""
from __future__ import annotations

import os
from datetime import timedelta

from dotenv import load_dotenv
from livekit import api

load_dotenv()
# `livekit-server --dev` defaults; the app's minter has none of its own.
LIVEKIT_URL = os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.setdefault("LIVEKIT_API_SECRET", "secret")
TEST_EMAIL = os.environ.get("VOICE_TEST_EMAIL", "tester@example.com")
# This process mints for the test user only, so voice is on for it here.
os.environ.setdefault("FINNIE_VOICE_ENABLED", "1")
os.environ.setdefault("FINNIE_VOICE_ALLOWED_EMAILS", TEST_EMAIL)

from src.voice.access import new_voice_room, token_metadata  # noqa: E402
from src.voice.tokens import mint_call_token  # noqa: E402


def local_identity(name: str) -> str:
    return f"g-local-{''.join(c for c in name if c.isalnum()) or 'tester'}"


def mint_token(identity: str, room: str, email: str = TEST_EMAIL) -> str:
    """Join one room, publish microphone audio only (design doc: token rules)."""
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=False,
        can_publish_sources=["microphone"],
    )
    return (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_metadata(token_metadata(email))
        .with_grants(grants)
        .with_ttl(timedelta(minutes=10))
        .to_jwt()
    )


def new_call(name: str) -> tuple[str, str, str]:
    """(identity, room, token) for one local call, from the app's minter."""
    identity = local_identity(name)
    call = mint_call_token(identity, TEST_EMAIL)
    return identity, call.room, call.token


def raw_call(name: str, email: str) -> tuple[str, str, str]:
    """(identity, room, token) signed directly with `email`, skipping the minter."""
    identity = local_identity(name)
    room = new_voice_room(identity)
    return identity, room, mint_token(identity, room, email)
