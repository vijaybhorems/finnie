"""Imported once in LiveKit's forkserver, so a new call process starts without the heavy imports.

Each call runs in its own process, and when a call takes one of the idle
processes (`voice.idle_processes`), LiveKit starts a replacement at once, on
the same CPUs as the call that just began. Importing torch, sentence-transformers,
LangChain, LangGraph and the agents cost that replacement about 5 s of CPU; while
it ran, the new call's VAD fell behind ("VAD inference is slower than realtime")
and the worker reported itself full.

On Linux (the worker pool), LiveKit forks job processes from a forkserver and
imports `AgentServer(preload_modules=...)` there once; every process forked
afterwards inherits these modules copy-on-write, so its prewarm only loads the
embedding model and opens its pools. Under `spawn` (macOS) nothing changes:
each process imports what it needs itself, as before.

Only imports belong here: no model weights, threads, sockets or event loops,
which a forked process can't safely inherit. The forkserver stops at the first
exception other than ImportError, so nothing here may raise.
"""
from __future__ import annotations

import importlib
import time

# torch + transformers alone are ~3.5 s of the ~5 s.
HEAVY = (
    "torch",
    "sentence_transformers",
)
# LangChain, LangGraph, the Anthropic SDK, pandas/yfinance and Finnie's own modules,
# in roughly the order the worker imports them.
APP = (
    "src.core.config",
    "src.core.embeddings",
    "src.workflow.graph",
    "src.persistence.backend",
    "src.persistence.user_data",
    "langgraph.checkpoint.postgres",
    "langgraph.store.postgres",
    "psycopg_pool",
    "src.rag.retriever",
    "src.rag.pgvector",
    "src.memory.service",
    "src.web_app.warmup",
    "src.voice.prefetch",
    "src.voice.turns",
    "src.voice.usage",
)


def _tracing_on() -> bool:
    """Phoenix tracing patches LangChain as it is imported, so it must come first (src/core/tracing.py)."""
    try:
        from src.core.config import get_settings

        tracing = get_settings().tracing
        return bool(tracing.enabled and tracing.endpoint)
    except Exception:  # noqa: BLE001 — unknown: assume on, and leave LangChain to the job process
        return True


def preload() -> dict[str, object]:
    """Import what a call process needs; never raises. Returns what was imported and how long it took."""
    started = time.perf_counter()
    names = HEAVY + (() if _tracing_on() else APP)
    failed = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — the job process imports (and reports) it itself
            failed.append(name)
    return {"seconds": round(time.perf_counter() - started, 2), "modules": len(names), "failed": failed}


RESULT = preload()
