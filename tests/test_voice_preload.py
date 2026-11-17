"""src/voice/preload.py: imported in LiveKit's forkserver, so it must never raise or start threads."""
from __future__ import annotations

import importlib
import sys
import threading

import pytest

from src.voice import preload


def test_imports_the_heavy_modules_and_finnies_own():
    result = preload.preload()
    assert result["failed"] == []
    for name in preload.HEAVY + preload.APP:
        assert name in sys.modules


def test_a_failing_import_is_reported_not_raised(monkeypatch):
    real = importlib.import_module

    def flaky(name: str, *args, **kwargs):
        if name == "src.workflow.graph":
            raise RuntimeError("broken at import")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(preload.importlib, "import_module", flaky)
    result = preload.preload()
    assert result["failed"] == ["src.workflow.graph"]


def test_tracing_leaves_langchain_to_the_job_process(monkeypatch):
    """Phoenix must patch LangChain before it is imported; a forkserver import would come first."""
    monkeypatch.setattr(preload, "_tracing_on", lambda: True)
    imported: list[str] = []
    monkeypatch.setattr(preload.importlib, "import_module", lambda name: imported.append(name))
    preload.preload()
    assert imported == list(preload.HEAVY)


@pytest.mark.parametrize("enabled,endpoint,on", [(False, "", False), (True, "", False), (True, "http://x:4317", True)])
def test_tracing_on_needs_both_flag_and_endpoint(monkeypatch, enabled, endpoint, on):
    from src.core.config import get_settings

    tracing = get_settings().tracing
    monkeypatch.setattr(tracing, "enabled", enabled)
    monkeypatch.setattr(tracing, "endpoint", endpoint)
    assert preload._tracing_on() is on


def test_starts_no_threads():
    """A forked process inherits no threads; one started here would be missing in every call process."""
    before = threading.active_count()
    preload.preload()
    assert threading.active_count() == before
