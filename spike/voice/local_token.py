"""Tokens for the local test tools (talk page, scripted caller). LOCAL USE ONLY.

They pass the worker's join checks (src/voice/access.py) as a made-up Google
user: identity g-local-<name>, a room named after it, and VOICE_TEST_EMAIL in
the metadata. Start the worker with voice on and that email allowed:
    FINNIE_VOICE_ENABLED=1 FINNIE_VOICE_ALLOWED_EMAILS=tester@example.com \
        .venv-voice/bin/python -m src.voice.worker dev
The production minter (rate limits, one call per user) comes with the token
endpoint (design doc: Token rules).
"""
from __future__ import annotations

import os
from datetime import timedelta

from dotenv import load_dotenv
from livekit import api

from src.voice.access import new_voice_room, token_metadata

load_dotenv()

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
TEST_EMAIL = os.environ.get("VOICE_TEST_EMAIL", "tester@example.com")


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


def new_call(name: str, email: str = TEST_EMAIL) -> tuple[str, str, str]:
    """(identity, room, token) for one local call."""
    identity = local_identity(name)
    room = new_voice_room(identity)
    return identity, room, mint_token(identity, room, email)
