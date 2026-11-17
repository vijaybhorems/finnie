"""src/voice/preload.py: imported in LiveKit's forkserver, so it must never raise or start threads."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
from pathlib import Path

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


_SPANS_AFTER_PRELOAD = """
import sys
import src.voice.preload  # as in the forkserver: LangChain and LangGraph are imported first
assert "langchain_core" in sys.modules and "langgraph" in sys.modules
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

spans = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(spans))
LangChainInstrumentor().instrument(tracer_provider=provider)  # what setup_tracing() runs, later

from typing import TypedDict
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    q: str
    a: str

model = FakeListChatModel(responses=["hi"])
graph = StateGraph(State)
graph.add_node("agent", lambda s: {"a": model.invoke(s["q"]).content})
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
graph.compile().invoke({"q": "x"})
print(sorted({span.name for span in spans.get_finished_spans()}))
"""


def test_tracing_still_records_spans_when_langchain_was_preloaded():
    """Production runs Phoenix tracing, which starts in the job process after the forkserver's imports."""
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-c", _SPANS_AFTER_PRELOAD], cwd=root, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(root)}, timeout=180)
    assert done.returncode == 0, done.stderr[-2000:]
    names = done.stdout.strip().splitlines()[-1]
    for span in ("LangGraph", "agent", "FakeListChatModel"):
        assert span in names


def test_starts_no_threads():
    """A forked process inherits no threads; one started here would be missing in every call process."""
    before = threading.active_count()
    preload.preload()
    assert threading.active_count() == before
