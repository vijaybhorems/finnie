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

        register(
            project_name=tracing.project_name,
            endpoint=tracing.endpoint,
            auto_instrument=True,  # picks up installed OpenInference instrumentors
            batch=True,            # batched export for production
        )
        _TRACING_INITIALISED = True
        logger.info(
            "tracing_enabled",
            project_name=tracing.project_name,
            endpoint=tracing.endpoint,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — tracing must never break the app
        logger.warning("tracing_setup_failed", error=str(exc))
        return False
