#!/usr/bin/env python3
"""Convenience script to build or rebuild the FAISS knowledge index."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # must precede src imports

# pylint: disable=wrong-import-position
from src.rag.indexer import RAGIndexer
from src.utils.logger import get_logger, setup_logging
# pylint: enable=wrong-import-position

setup_logging()
logger = get_logger(__name__)


def main():
    """Parse CLI arguments and run the RAG index build."""
    parser = argparse.ArgumentParser(description="Build the Finnie RAG knowledge index")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index already exists")
    args = parser.parse_args()

    logger.info("starting_rag_index_build", force=args.force)
    indexer = RAGIndexer()
    indexer.build_index(force=args.force)
    logger.info("rag_index_build_complete")


if __name__ == "__main__":
    main()
