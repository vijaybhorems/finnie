"""Arize Phoenix / OpenInference tracing setup.

Auto-instruments LangChain + LangGraph (the LangChain instrumentor covers
LangGraph nodes and ``langchain-anthropic``) and exports spans to a Phoenix
collector.

IMPORTANT: ``setup_tracing()`` must run BEFORE any LangChain/LangGraph module is
imported so the instrumentor can patch them. Call it at the very start of the
process entry point (see ``src/web_app/app.py``).

Tracing is a strict no-op unless ``tracing.enabled`` is true AND an endpoint is
configured, so the app never hard-depends on a running Phoenix instance.
"""
from __future__ import annotations

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRACING_INITIALISED = False


def setup_tracing() -> bool:
    """Initialise Phoenix tracing if enabled. Returns True if instrumentation is active.

    Safe to call multiple times; only the first successful call registers spans.
    """
    global _TRACING_INITIALISED
    if _TRACING_INITIALISED:
        return True

    settings = get_settings()
    tracing = settings.tracing

    if not tracing.enabled or not tracing.endpoint:
        logger.info(
            "tracing_disabled",
            enabled=tracing.enabled,
            has_endpoint=bool(tracing.endpoint),
        )
        return False

    try:
        from phoenix.otel import register

        # Protocol: explicit config wins; otherwise infer from the endpoint so both
        # local dev (gRPC collector on :4317) and Phoenix Cloud / any HTTPS-fronted
        # collector (OTLP over HTTP, e.g. Cloud Run) work without extra config.
        protocol = tracing.protocol or (
            "http/protobuf" if tracing.endpoint.startswith("https://") else "grpc"
        )

        # register(endpoint=...) expects a fully-qualified OTLP URL and uses it
        # verbatim (unlike the PHOENIX_COLLECTOR_ENDPOINT env var, which Phoenix
        # normalizes). For HTTP we must append the /v1/traces path ourselves or
        # spans POST to the space root and fail with 405 Method Not Allowed.
        endpoint = tracing.endpoint
        if protocol == "http/protobuf" and not endpoint.rstrip("/").endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"

        register_kwargs: dict = dict(
            project_name=tracing.project_name,
            endpoint=endpoint,
            protocol=protocol,
            auto_instrument=True,  # picks up installed OpenInference instrumentors
            batch=True,            # batched export for production
        )
        if tracing.api_key:
            register_kwargs["api_key"] = tracing.api_key

        register(**register_kwargs)
        _TRACING_INITIALISED = True
        logger.info(
            "tracing_enabled",
            project_name=tracing.project_name,
            endpoint=endpoint,
            protocol=protocol,
            has_api_key=bool(tracing.api_key),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — tracing must never break the app
        logger.warning("tracing_setup_failed", error=str(exc))
        return False
