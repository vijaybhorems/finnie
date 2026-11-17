"""The voice worker's env vars and secrets, taken from the web app's Cloud Run service.

The worker runs the same graph as the app (same Anthropic and data-provider
keys, same database), and re-checks the voice flag and allowlist on every call,
so those must match the app's. Rather than keep a second list in cloudbuild.yaml,
the deploy step copies them from the app service on every deploy:

    gcloud run services describe finnie-app --region=us-central1 --format=json > app.json
    python3 deploy/worker_env.py env app.json       # for --update-env-vars
    python3 deploy/worker_env.py secrets app.json   # for --update-secrets

Secrets are copied as references (secret name and version), never values. The
app's sign-in secrets are not copied: the worker has no use for them. Speech
keys exist only for the worker and are referenced by name.

Standard library only: it runs in Cloud Build's cloud-sdk image.
"""
from __future__ import annotations

import json
import sys

# Plain env vars copied when the app has them. FINNIE_VOICE_* must match: the app
# mints tokens by them and the worker checks every call against its own copy.
MIRRORED_ENV = [
    "FINNIE_VOICE_ENABLED", "FINNIE_VOICE_ALLOWED_EMAILS",
    "REDIS_HOST", "REDIS_PORT", "LOG_LEVEL", "TRACING_ENABLED", "PHOENIX_COLLECTOR_ENDPOINT",
]
# Secrets copied from the app's references. The first group must be there.
REQUIRED_SECRETS = ["DATABASE_URL", "ANTHROPIC_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
OPTIONAL_SECRETS = ["ALPHA_VANTAGE_API_KEY", "FRED_API_KEY", "NEWS_API_KEY", "PHOENIX_API_KEY"]
# Only the worker uses these: {env var the worker reads: secret deploy/setup_gcp.sh
# stores}. LiveKit's ElevenLabs plugin reads ELEVEN_API_KEY and nothing else.
WORKER_ONLY_SECRETS = {"DEEPGRAM_API_KEY": "DEEPGRAM_API_KEY", "ELEVEN_API_KEY": "ELEVENLABS_API_KEY"}


def _env_entries(service: dict) -> list[dict]:
    try:
        return service["spec"]["template"]["spec"]["containers"][0].get("env", [])
    except (KeyError, IndexError, TypeError):
        raise SystemExit("not a Cloud Run service description (gcloud run services describe --format=json)")


def env_vars(service: dict) -> str:
    """`--update-env-vars` value. Separated by '|' (gcloud's ^|^ form): the allowlist
    holds commas, and its emails hold '@'."""
    plain = {e["name"]: e.get("value", "") for e in _env_entries(service) if "valueFrom" not in e}
    pairs = [f"{name}={plain[name]}" for name in MIRRORED_ENV if name in plain]
    if "FINNIE_VOICE_ENABLED" not in plain:
        pairs.append("FINNIE_VOICE_ENABLED=0")  # spelled out: the app has voice off
    if any("|" in pair for pair in pairs):
        raise SystemExit("an env var copied to the worker contains '|', the separator")
    return "^|^" + "|".join(pairs)


def secrets(service: dict) -> str:
    """`--update-secrets` value: the app's references, plus the worker's own."""
    refs = {}
    for entry in _env_entries(service):
        ref = entry.get("valueFrom", {}).get("secretKeyRef")
        if ref:
            refs[entry["name"]] = f"{ref['name']}:{ref.get('key', 'latest')}"
    missing = [name for name in REQUIRED_SECRETS if name not in refs]
    if missing:
        raise SystemExit(
            "the app service has no " + ", ".join(missing)
            + ": run deploy/setup_gcp.sh (Voice stage) to add them, or deploy with _VOICE=0"
        )
    pairs = [f"{name}={refs[name]}" for name in REQUIRED_SECRETS + OPTIONAL_SECRETS if name in refs]
    pairs += [f"{env}={secret}:latest" for env, secret in WORKER_ONLY_SECRETS.items()]
    return ",".join(pairs)


def main(argv: list[str]) -> None:
    if len(argv) != 3 or argv[1] not in ("env", "secrets"):
        raise SystemExit("usage: worker_env.py env|secrets SERVICE_JSON")
    with open(argv[2]) as f:
        service = json.load(f)
    print(env_vars(service) if argv[1] == "env" else secrets(service))


if __name__ == "__main__":
    main(sys.argv)
