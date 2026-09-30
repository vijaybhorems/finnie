"""Persistence backends shared by the whole process.

One LangGraph checkpointer (conversation threads) and one LangGraph store
(profile, holdings, saved plan), both on a single Postgres connection pool when
DATABASE_URL is set, or in-process memory otherwise.

The memory backend keeps the app runnable without a database — the behaviour
before persistence existed — but nothing survives a restart and nothing is
shared between Cloud Run instances. A warning is logged once so that is never
mistaken for a working deployment.
"""
from __future__ import annotations

import atexit
from functools import lru_cache
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PersistenceConfigError(RuntimeError):
    """Persistence is configured in a way that cannot work."""


def backend_name() -> str:
    """"postgres" or "memory", resolving the "auto" setting."""
    settings = get_settings()
    backend = settings.persistence.backend
    if backend == "auto":
        return "postgres" if settings.database_url else "memory"
    if backend == "postgres" and not settings.database_url:
        raise PersistenceConfigError("persistence.backend is 'postgres' but DATABASE_URL is not set")
    return backend


@lru_cache(maxsize=1)
def get_pool() -> Any:
    """The process-wide Postgres connection pool, opened and verified."""
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    settings = get_settings()
    # Connection settings required by LangGraph's Postgres checkpointer and store.
    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=settings.persistence.pool_max_size,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
        name="finnie",
    )
    # Fail at startup rather than on a user's first message. With persistence
    # configured, an unreachable database is a deployment error — silently
    # falling back to memory would discard every profile saved until noticed.
    pool.open(wait=True, timeout=settings.persistence.connect_timeout_seconds)
    # Close at interpreter exit. Otherwise psycopg_pool's finaliser waits up to
    # 5s for each worker thread to stop — ~20s added to every shutdown (a CLI
    # run, or Cloud Run's SIGTERM grace period).
    atexit.register(pool.close)
    logger.info("postgres_pool_open", max_size=settings.persistence.pool_max_size)
    return pool


def _warn_memory_backend(component: str) -> None:
    logger.warning(
        "persistence_in_memory",
        component=component,
        detail="DATABASE_URL not set: user data and chat history are lost on restart",
    )


@lru_cache(maxsize=1)
def get_checkpointer() -> BaseCheckpointSaver:
    """Conversation-thread checkpointer."""
    if backend_name() == "memory":
        _warn_memory_backend("checkpointer")
        return InMemorySaver()

    from langgraph.checkpoint.postgres import PostgresSaver

    saver = PostgresSaver(get_pool())
    saver.setup()  # idempotent: creates or migrates the checkpoint tables
    return saver


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embedding function for the store's semantic index (the shared MiniLM model).

    A module-level function so tests can substitute a deterministic embedder
    before the store is built.
    """
    from src.core.embeddings import embed

    return embed(texts).tolist()


def _index_config() -> dict[str, Any]:
    """Semantic index over memories (items with a "fact" field).

    "flat" = exact search, no ANN index (LangGraph's default is HNSW). Every
    user's memories share one table, filtered by namespace. An approximate
    index could in principle rank globally and filter afterwards, dropping a
    user's own memories; with LangGraph's current query (namespace filter on
    the store's primary key, joined to the vectors) the planner doesn't do
    that — we checked, including with sequential scans disabled. Exact search
    guarantees it whatever plan is chosen, and costs nothing at per-user
    volumes (a user has at most `memory.max_memories_per_user` rows).
    """
    return {
        "dims": get_settings().embeddings.dimension,
        "embed": lambda texts: embed_texts(texts),
        "fields": ["fact"],
        "ann_index_config": {"kind": "flat"},
    }


@lru_cache(maxsize=1)
def get_store() -> BaseStore:
    """Key-value store for per-user data, with a semantic index for memories."""
    if backend_name() == "memory":
        _warn_memory_backend("store")
        return InMemoryStore(index=_index_config())

    from langgraph.store.postgres import PostgresStore

    store = PostgresStore(get_pool(), index=_index_config())
    store.setup()  # idempotent: creates or migrates the store (and vector) tables
    return store


def reset_persistence() -> None:
    """Drop the singletons and close the pool (tests, and config reloads)."""
    get_checkpointer.cache_clear()
    get_store.cache_clear()
    if get_pool.cache_info().currsize:
        get_pool().close()
    get_pool.cache_clear()
