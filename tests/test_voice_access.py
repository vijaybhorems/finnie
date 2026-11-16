"""src/voice/access.py: the worker's join checks, and voice settings."""
from __future__ import annotations

import pytest

from src.core.config import VoiceConfig, get_settings
from src.voice.access import check_join, is_allowed, new_voice_room, token_metadata

USER = "g-1234567890"
ALICE = token_metadata("Alice@Example.com")


def _voice(**kw) -> VoiceConfig:
    return VoiceConfig(**{"enabled": True, "allowed_emails": ["alice@example.com"], **kw})


class TestCheckJoin:
    def test_own_room_on_the_allowlist_joins(self):
        assert check_join(USER, new_voice_room(USER), ALICE, _voice()) is None

    def test_disabled_refuses_everyone(self):
        assert check_join(USER, new_voice_room(USER), ALICE, _voice(enabled=False)) == "voice_disabled"

    @pytest.mark.parametrize("identity", ["", "spike-x", "e-abc", "g-", "g-12.3", "g-1:2", "G-123"])
    def test_identity_must_be_a_google_user_id(self, identity):
        assert check_join(identity, f"{identity}:{'a' * 32}", ALICE, _voice()) == "bad_identity"

    @pytest.mark.parametrize("room", [
        f"g-999:{'a' * 32}",          # someone else's room
        f"{USER}x:{'a' * 32}",        # a longer id sharing the prefix
        f"{USER}:{'a' * 31}",         # malformed suffix
        f"{USER}:{'A' * 32}",
        f"{USER}:{'a' * 32}:extra",
        USER,
        "",
    ])
    def test_room_must_be_the_users_own(self, room):
        assert check_join(USER, room, ALICE, _voice()) == "room_not_owned"

    @pytest.mark.parametrize("metadata", ["", "not json", "[]", '{"email": 5}', token_metadata("bob@example.com")])
    def test_email_must_be_on_the_allowlist(self, metadata):
        assert check_join(USER, new_voice_room(USER), metadata, _voice()) == "not_allowlisted"

    def test_empty_allowlist_lets_nobody_in(self):
        assert check_join(USER, new_voice_room(USER), ALICE, _voice(allowed_emails=[])) == "not_allowlisted"

    def test_star_lets_every_signed_in_user_in(self):
        voice = _voice(allowed_emails=["*"])
        assert check_join(USER, new_voice_room(USER), token_metadata("bob@example.com"), voice) is None
        assert not is_allowed("", voice)


def test_new_voice_room_is_fresh_and_needs_a_google_id():
    a, b = new_voice_room(USER), new_voice_room(USER)
    assert a != b and a.startswith(f"{USER}:")
    with pytest.raises(ValueError):
        new_voice_room("e-abc")


class TestVoiceSettings:
    def test_off_by_default(self):
        voice = get_settings().voice
        assert voice.enabled is False and voice.allowed_emails == []
        assert voice.db_pool_max_size <= 2

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("FINNIE_VOICE_ENABLED", "true")
        monkeypatch.setenv("FINNIE_VOICE_ALLOWED_EMAILS", " Alice@Example.com , ,bob@example.com")
        get_settings.cache_clear()
        voice = get_settings().voice
        assert voice.enabled is True
        assert voice.allowed_emails == ["alice@example.com", "bob@example.com"]
        assert is_allowed("ALICE@example.com")
