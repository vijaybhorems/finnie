"""Background warm-up of the heavy singletons, off the page-render path.

Importing LangChain/LangGraph and loading the sentence-transformers model takes
~10s on a laptop and over a minute on a cold Cloud Run instance. Doing it inside
the page script means the first visitor stares at a blank page (and then a
spinner) while it happens. Instead it runs once per process in a daemon thread:
started by ``src.web_app.serve`` as the container boots, or by the first page
render if the app was launched with plain ``streamlit run``. The login page never
waits for it; a signed-in page calls ``wait()`` only when it needs the workflow.

This module's state lives in ``sys.modules``, so it survives Streamlit reruns
(which re-execute ``app.py`` but not the modules it imports).
"""
from __future__ import annotations

import threading
import time

from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

_lock = threading.Lock()
_thread: threading.Thread | None = None
_done = threading.Event()
_tracing_done = threading.Event()


def _run() -> None:
    started = time.monotonic()
    try:
        setup_logging()
        # Tracing must be registered before LangChain/LangGraph are imported so
        # the OpenInference instrumentor can patch them — so it runs here, first.
        from src.core.tracing import setup_tracing

        try:
            setup_tracing()
        finally:
            _tracing_done.set()

        from src.workflow.graph import build_graph

        build_graph()
        try:
            from src.rag.retriever import get_retriever

            get_retriever().warm_up()
        except Exception as exc:  # non-fatal: RAG falls back to lazy load on first query
            logger.warning("warmup_retriever_failed", error=str(exc))
        logger.info("startup_warmup_complete", seconds=round(time.monotonic() - started, 1))
    except Exception as exc:  # never kill the process; pages import lazily anyway
        logger.warning("startup_warmup_failed", error=str(exc))
    finally:
        _tracing_done.set()
        _done.set()


def start() -> None:
    """Start the warm-up thread if it isn't running or finished. Idempotent."""
    global _thread
    with _lock:
        if _thread is None:
            _thread = threading.Thread(target=_run, name="finnie-warmup", daemon=True)
            _thread.start()


def is_ready() -> bool:
    return _done.is_set()


def wait(timeout: float | None = None) -> bool:
    """Block until warm-up has finished (successfully or not)."""
    start()
    return _done.wait(timeout)


def wait_for_tracing(timeout: float | None = None) -> bool:
    """Block until tracing is registered, so a caller may import LangChain safely."""
    start()
    return _tracing_done.wait(timeout)
