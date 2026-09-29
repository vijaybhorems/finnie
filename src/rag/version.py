"""Knowledge-base content version.

The FAQ cache stores answers grounded in knowledge-base articles. When an
article changes, every cached answer derived from it must stop being served —
so entries are stamped with this hash and ignored when it moves.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_kb_version() -> str:
    """Return a short hash over every knowledge-base file's content."""
    kb_path = Path(get_settings().knowledge_base_path)
    digest = hashlib.sha256()

    if kb_path.exists():
        for file_path in sorted(kb_path.rglob("*")):
            if file_path.suffix not in (".txt", ".md") or not file_path.is_file():
                continue
            digest.update(str(file_path.relative_to(kb_path)).encode("utf-8"))
            digest.update(file_path.read_bytes())

    version = digest.hexdigest()[:16]
    logger.info("kb_version_computed", version=version)
    return version
