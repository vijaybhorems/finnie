"""deploy/worker_env.py: the voice worker's config, copied from the app's Cloud Run service."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "worker_env.py"
_spec = importlib.util.spec_from_file_location("worker_env", SCRIPT)
worker_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker_env)


def _service(env: list[dict]) -> dict:
    return {"spec": {"template": {"spec": {"containers": [{"env": env}]}}}}


def _secret(name: str, secret: str | None = None, version: str = "latest") -> dict:
    return {"name": name, "valueFrom": {"secretKeyRef": {"name": secret or name, "key": version}}}


APP = _service([
    {"name": "ALLOWED_EMAILS", "value": "a@example.com,b@example.com"},
    {"name": "AUTH_REDIRECT_URI", "value": "https://finnie.example/oauth2callback"},
    {"name": "FINNIE_VOICE_ENABLED", "value": "1"},
    {"name": "FINNIE_VOICE_ALLOWED_EMAILS", "value": "a@example.com,b@example.com"},
    {"name": "LOG_LEVEL", "value": "INFO"},
    _secret("DATABASE_URL"),
    _secret("ANTHROPIC_API_KEY", "anthropic-prod", "3"),
    _secret("FRED_API_KEY"),
    _secret("GOOGLE_CLIENT_SECRET"),
    _secret("AUTH_COOKIE_SECRET"),
    _secret("LIVEKIT_URL"),
    _secret("LIVEKIT_API_KEY"),
    _secret("LIVEKIT_API_SECRET"),
])


def test_the_worker_gets_the_apps_voice_flag_and_allowlist():
    assert worker_env.env_vars(APP) == (
        "^|^FINNIE_VOICE_ENABLED=1|FINNIE_VOICE_ALLOWED_EMAILS=a@example.com,b@example.com|LOG_LEVEL=INFO"
    )


def test_sign_in_settings_stay_with_the_app():
    env, secrets = worker_env.env_vars(APP), worker_env.secrets(APP)
    for name in ("ALLOWED_EMAILS=", "AUTH_REDIRECT_URI", "GOOGLE_CLIENT_SECRET", "AUTH_COOKIE_SECRET"):
        assert name not in env.replace("FINNIE_VOICE_ALLOWED_EMAILS", "") and name not in secrets


def test_secrets_are_copied_as_references_plus_the_speech_keys():
    assert worker_env.secrets(APP).split(",") == [
        "DATABASE_URL=DATABASE_URL:latest",
        "ANTHROPIC_API_KEY=anthropic-prod:3",  # the app's own secret name and pinned version
        "LIVEKIT_URL=LIVEKIT_URL:latest",
        "LIVEKIT_API_KEY=LIVEKIT_API_KEY:latest",
        "LIVEKIT_API_SECRET=LIVEKIT_API_SECRET:latest",
        "FRED_API_KEY=FRED_API_KEY:latest",
        "DEEPGRAM_API_KEY=DEEPGRAM_API_KEY:latest",
        "ELEVEN_API_KEY=ELEVENLABS_API_KEY:latest",
    ]


def test_speech_keys_reach_the_worker_under_the_names_it_reads_and_redacts():
    """LiveKit's plugins read DEEPGRAM_API_KEY and ELEVEN_API_KEY; the worker's logs mask
    exactly those (src/utils/redact.py). A key under another name leaves the voice silent."""
    from src.utils.redact import _SECRET_ENV_VARS

    assert set(worker_env.WORKER_ONLY_SECRETS) <= set(_SECRET_ENV_VARS)


def test_an_app_without_voice_keeps_the_worker_off():
    app = _service([_secret(n) for n in worker_env.REQUIRED_SECRETS])
    assert worker_env.env_vars(app) == "^|^FINNIE_VOICE_ENABLED=0"


def test_missing_livekit_keys_stop_the_deploy_with_a_reason():
    app = _service([_secret("DATABASE_URL"), _secret("ANTHROPIC_API_KEY")])
    with pytest.raises(SystemExit, match="LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.*setup_gcp.sh"):
        worker_env.secrets(app)


def test_a_separator_in_a_copied_value_is_refused():
    app = _service([{"name": "FINNIE_VOICE_ALLOWED_EMAILS", "value": "a@example.com|b@example.com"}])
    with pytest.raises(SystemExit, match="separator"):
        worker_env.env_vars(app)


def test_runs_as_cloud_build_calls_it(tmp_path):
    path = tmp_path / "app.json"
    path.write_text(json.dumps(APP))
    out = subprocess.run([sys.executable, str(SCRIPT), "secrets", str(path)],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == worker_env.secrets(APP)
    bad = subprocess.run([sys.executable, str(SCRIPT), "secrets", str(tmp_path / "missing.json")],
                         capture_output=True, text=True)
    assert bad.returncode != 0


def _with_annotations(service: dict, **annotations: str) -> dict:
    service = json.loads(json.dumps(service))
    service["spec"]["template"]["metadata"] = {
        "annotations": {f"run.googleapis.com/{k.replace('_', '-')}": v for k, v in annotations.items()}
    }
    return service


def test_the_worker_uses_the_apps_vpc_connector_for_private_addresses_only():
    """REDIS_HOST is a private Memorystore address; the app reaches it through a connector.
    The worker takes the same route, but only for private ranges, even when the app sends
    everything: call audio and the speech vendors keep Cloud Run's internet egress."""
    app = _with_annotations(APP, vpc_access_connector="finnie-vpc-connector", vpc_access_egress="all-traffic")
    assert worker_env.network(app) == ["--vpc-connector=finnie-vpc-connector", "--vpc-egress=private-ranges-only"]


def test_the_worker_mirrors_direct_vpc_egress():
    nics = '[{"network":"default","subnetwork":"default","tags":["redis-client"]}]'
    app = _with_annotations(APP, network_interfaces=nics)
    assert worker_env.network(app) == [
        "--network=default", "--subnet=default", "--network-tags=redis-client", "--vpc-egress=private-ranges-only",
    ]


def test_an_app_without_a_vpc_route_gives_the_worker_none():
    assert worker_env.network(APP) == []
    assert worker_env.network(_with_annotations(APP, execution_environment="gen2")) == []


def test_network_flags_print_one_per_line(tmp_path):
    path = tmp_path / "app.json"
    path.write_text(json.dumps(_with_annotations(APP, vpc_access_connector="finnie-vpc-connector")))
    out = subprocess.run([sys.executable, str(SCRIPT), "network", str(path)],
                         capture_output=True, text=True, check=True).stdout
    assert out.split() == ["--vpc-connector=finnie-vpc-connector", "--vpc-egress=private-ranges-only"]
