"""Keep API keys out of logs and out of error text that reaches prompts.

FRED and Alpha Vantage take their key as a query parameter, and `requests`
puts the full URL in an HTTPError's message. Logging `str(exc)`, or returning
it as an `error` value that an agent then sends to the model, leaks the key.

- `describe_request_error()`: a short, key-free description for error values
  that may be shown to the model or a user ("HTTP 400 Bad Request").
- `redact()`: the full message with known secrets and `api_key=`-style query
  values masked, for logs.
- `redact_processor`: a structlog processor that masks every configured secret
  in every log event, as a backstop for call sites that forget.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable

import requests

MASK = "[REDACTED]"
_MIN_SECRET_LEN = 8  # shorter values (e.g. LiveKit's "secret" dev default) would mask ordinary words
_QUERY_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|token|secret)=[^&\s'\"<>]+"
)
# Secrets not held on Settings: the voice worker's vendor keys and OAuth.
_SECRET_ENV_VARS = (
    "DEEPGRAM_API_KEY",
    "ELEVEN_API_KEY",
    "LIVEKIT_API_SECRET",
    "GOOGLE_CLIENT_SECRET",
)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """`text` with each secret and every `api_key=`-style query value masked."""
    for secret in secrets:
        if secret and len(secret) >= _MIN_SECRET_LEN:
            text = text.replace(secret, MASK)
    return _QUERY_SECRET.sub(lambda m: f"{m.group(1)}={MASK}", text)


def describe_request_error(exc: BaseException, secrets: Iterable[str] = ()) -> str:
    """A short description of a failed HTTP call, never containing the URL."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        reason = exc.response.reason or ""
        return f"HTTP {exc.response.status_code} {reason}".strip()
    if isinstance(exc, requests.Timeout):
        return "request timed out"
    if isinstance(exc, requests.ConnectionError):
        return "connection error"
    return redact(f"{type(exc).__name__}: {exc}", secrets)


def configured_secrets() -> list[str]:
    """Every secret this process knows, from Settings and the environment."""
    from src.core.config import get_settings

    settings = get_settings()
    values = [
        settings.anthropic_api_key,
        settings.alpha_vantage_api_key,
        settings.fred_api_key,
        settings.news_api_key,
        settings.phoenix_api_key,
        *(os.environ.get(name, "") for name in _SECRET_ENV_VARS),
    ]
    return [v for v in values if v and len(v) >= _MIN_SECRET_LEN]


def make_redact_processor(secrets: Iterable[str]):
    """A structlog processor masking `secrets` in every string in the event."""
    known = tuple(secrets)

    def _clean(value: Any) -> Any:
        if isinstance(value, str):
            return redact(value, known)
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(_clean(v) for v in value)
        return value

    def redact_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return {k: _clean(v) for k, v in event_dict.items()}

    return redact_processor
