"""Sync the knowledge base into Postgres/pgvector.

    python -m src.rag.sync

Idempotent: embeds only new or changed chunks and deletes chunks whose files
are gone. The app also runs this at startup (PgVectorRetriever.warm_up), so an
edited article is live after the next deploy with no separate rebuild step;
run it by hand to apply an edit without restarting, or as a deploy step.
"""
from __future__ import annotations

import sys

from src.rag.pgvector import sync_knowledge_base
from src.utils.logger import setup_logging


def main() -> int:
    setup_logging()
    report = sync_knowledge_base()
    print(
        f"knowledge base synced: {report.inserted} inserted, {report.updated} updated, "
        f"{report.deleted} deleted, {report.unchanged} unchanged"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
