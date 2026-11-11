"""Structured logging setup."""
from __future__ import annotations

import logging
import sys

import structlog

from src.core.config import get_settings
from src.utils.redact import configured_secrets, make_redact_processor


def setup_logging() -> None:
    """Configure structlog with processors and log level from application settings."""
    settings = get_settings()
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            make_redact_processor(configured_secrets()),  # no API key ever reaches a log line
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structlog bound logger."""
    return structlog.get_logger(name)
