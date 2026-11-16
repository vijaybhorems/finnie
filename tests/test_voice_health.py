"""src/voice/health.py: vendor circuit breakers shared between calls."""
from __future__ import annotations

from src.voice.health import DEEPGRAM, ELEVENLABS, VendorHealth


class FakeStore:
    """The slice of src/utils/cache.Cache that VendorHealth uses; one dict = one Redis."""

    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ttl=None):
        self.data[key] = value

    def delete(self, key):
        self.data.pop(key, None)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _health(store=None, clock=None):
    return VendorHealth(store or FakeStore(), clock or Clock())  # threshold 5, recovery 60 s


def test_opens_after_five_failures_in_a_row():
    health = _health()
    for _ in range(4):
        health.failure(DEEPGRAM)
    assert not health.is_open(DEEPGRAM)
    health.failure(DEEPGRAM)
    assert health.is_open(DEEPGRAM) and not health.is_open(ELEVENLABS)


def test_state_is_shared_between_calls():
    store, clock = FakeStore(), Clock()
    for _ in range(5):
        _health(store, clock).failure(ELEVENLABS)  # a new process per call
    assert _health(store, clock).is_open(ELEVENLABS)


def test_success_resets_the_count():
    health = _health()
    for _ in range(4):
        health.failure(DEEPGRAM)
    health.success(DEEPGRAM)
    health.failure(DEEPGRAM)
    assert not health.is_open(DEEPGRAM)


def test_half_open_probe_closes_or_reopens():
    clock = Clock()
    health = _health(clock=clock)
    for _ in range(5):
        health.failure(DEEPGRAM)
    clock.now += 61
    assert not health.is_open(DEEPGRAM)  # the next call may try
    health.failure(DEEPGRAM)             # the probe failed
    assert health.is_open(DEEPGRAM)
    clock.now += 61
    health.success(DEEPGRAM)             # this probe worked
    health.failure(DEEPGRAM)
    assert not health.is_open(DEEPGRAM)


def test_unreadable_state_counts_as_healthy():
    store = FakeStore()
    store.data["finnie:voice:health:deepgram"] = "garbage"
    assert not _health(store).is_open(DEEPGRAM)
