"""Simulated calls reach one LiveKit server, worker and caller alike (tests/evals/sim_caller.py)."""
from __future__ import annotations

import pytest

from tests.evals import sim_caller

LOCAL = {"LIVEKIT_URL": "ws://localhost:7880", "LIVEKIT_API_KEY": "devkey", "LIVEKIT_API_SECRET": "secret"}
CLOUD = {"SIM_LIVEKIT_URL": "wss://finnie-test.livekit.cloud", "SIM_LIVEKIT_API_KEY": "APItest",
         "SIM_LIVEKIT_API_SECRET": "s3cret"}


@pytest.fixture
def env_file(monkeypatch):
    values: dict[str, str] = {}
    monkeypatch.setattr(sim_caller, "_env_file", lambda: values)
    for name in [*LOCAL, *CLOUD]:
        monkeypatch.delenv(name, raising=False)
    return values


def test_the_local_dev_server_by_default(env_file):
    assert sim_caller.livekit_target() == LOCAL


def test_never_the_apps_own_livekit_keys(env_file):
    """A sim worker registered with the app's project would be handed real users' calls."""
    env_file.update({"LIVEKIT_URL": "wss://finnie-prod.livekit.cloud", "LIVEKIT_API_KEY": "APIprod",
                     "LIVEKIT_API_SECRET": "prod-secret"})
    assert sim_caller.livekit_target() == LOCAL


def test_a_test_project_named_in_env_file(env_file):
    env_file.update(CLOUD)
    assert sim_caller.livekit_target() == {
        "LIVEKIT_URL": "wss://finnie-test.livekit.cloud", "LIVEKIT_API_KEY": "APItest",
        "LIVEKIT_API_SECRET": "s3cret"}


def test_the_environment_wins_over_env_file(env_file, monkeypatch):
    env_file.update(CLOUD)
    monkeypatch.setenv("SIM_LIVEKIT_URL", "wss://other.livekit.cloud")
    assert sim_caller.livekit_target()["LIVEKIT_URL"] == "wss://other.livekit.cloud"


def test_a_partial_project_is_refused(env_file):
    env_file.update({"SIM_LIVEKIT_URL": CLOUD["SIM_LIVEKIT_URL"]})
    with pytest.raises(RuntimeError, match="SIM_LIVEKIT_API_KEY, SIM_LIVEKIT_API_SECRET"):
        sim_caller.livekit_target()
