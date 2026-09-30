"""Bring the database up to date, then exit.

    python -m src.persistence.migrate

Creates or migrates the LangGraph checkpoint and store tables (including the
memory vector table), the pgvector knowledge-base and FAQ-cache tables, and
syncs the knowledge base. Every step is idempotent: a no-op run takes a few
milliseconds.

The container runs this before Streamlit starts (docker/entrypoint.sh), so a
new Cloud Run revision only receives traffic once its database is ready — and
if the database is unreachable the revision fails to start, leaving the
previous one serving. Instances starting together are serialised by a
session-level advisory lock.
"""
from __future__ import annotations

import sys
import time

from src.persistence.backend import backend_name, get_checkpointer, get_pool, get_store
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Distinct from the knowledge-base sync's lock key (src/rag/pgvector.py): that
# lock is taken on another pool connection while this one is held, and a
# shared key would deadlock.
_MIGRATE_LOCK_KEY = 0x46494E4E4D47  # "FINNMG"
_LOCK_POLL_SECONDS = 0.5
_LOCK_TIMEOUT_SECONDS = 300


def _acquire_lock(conn) -> None:
    """Take the migration lock by polling, never by blocking in a statement.

    LangGraph's store setup runs CREATE INDEX CONCURRENTLY, which waits for
    every open transaction in the database to finish. A waiter blocked inside
    `SELECT pg_advisory_lock(...)` *is* an open transaction, so it would wait
    for the lock holder while the holder's index build waited for it: a
    deadlock Postgres doesn't detect. `pg_try_advisory_lock` returns at once,
    and between polls the waiter holds no transaction.
    """
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        if conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (_MIGRATE_LOCK_KEY,)).fetchone()["ok"]:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"another instance held the migration lock for over {_LOCK_TIMEOUT_SECONDS}s"
            )
        time.sleep(_LOCK_POLL_SECONDS)


def migrate() -> None:
    if backend_name() != "postgres":
        print("DATABASE_URL not set: nothing to migrate (in-memory backend).")
        return

    from src.rag.pgvector import ensure_schema, sync_knowledge_base

    started = time.monotonic()
    with get_pool().connection() as lock_conn:
        _acquire_lock(lock_conn)
        try:
            get_checkpointer()      # LangGraph checkpoint tables
            get_store()             # LangGraph store + memory vector tables
            ensure_schema()         # pgvector extension, kb_chunks, faq_cache
            report = sync_knowledge_base()
        finally:
            lock_conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATE_LOCK_KEY,))

    elapsed = time.monotonic() - started
    logger.info("migrate_complete", seconds=round(elapsed, 2), **report.__dict__)
    print(
        f"database ready in {elapsed:.1f}s — knowledge base: {report.inserted} inserted, "
        f"{report.updated} updated, {report.deleted} deleted, {report.unchanged} unchanged"
    )


def main() -> int:
    setup_logging()
    migrate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
