"""Tests for the concurrent fetch helper."""
from __future__ import annotations

import time

from src.utils.parallel import gather


class TestGather:
    def test_returns_all_results_by_key(self):
        results = gather({"a": lambda: 1, "b": lambda: "two"})
        assert results == {"a": 1, "b": "two"}

    def test_empty_tasks(self):
        assert gather({}) == {}

    def test_failing_task_yields_none_without_failing_others(self):
        def boom():
            raise RuntimeError("provider down")

        results = gather({"ok": lambda: "value", "bad": boom})
        assert results["ok"] == "value"
        assert results["bad"] is None

    def test_runs_concurrently(self):
        """Three 100 ms sleeps must take ~100 ms, not ~300 ms."""
        started = time.monotonic()
        gather({f"t{i}": (lambda: time.sleep(0.1)) for i in range(3)})
        elapsed = time.monotonic() - started
        assert elapsed < 0.25
