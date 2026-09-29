"""Run independent I/O-bound calls concurrently.

Agents fetch several independent things per turn (RAG context, macro data,
prices, sector performance). Each is a blocking HTTP/disk call, so running them
in a thread pool turns a sum of latencies into a max.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def gather(
    tasks: dict[str, Callable[[], Any]],
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Run every callable in `tasks` concurrently and return their results by key.

    A task that raises (or times out) yields ``None`` for its key and a warning
    log — callers already handle missing data, and one dead provider must not
    take down the turn.
    """
    if not tasks:
        return {}

    results: dict[str, Any] = {key: None for key in tasks}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {key: pool.submit(fn) for key, fn in tasks.items()}
        for key, future in futures.items():
            try:
                results[key] = future.result(timeout=timeout)
            except Exception as exc:  # noqa: BLE001 — one provider must not fail the turn
                logger.warning("parallel_task_failed", task=key, error=str(exc))
    return results
