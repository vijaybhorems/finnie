"""Container entry point: warm up before Streamlit opens the port, when asked to."""
from __future__ import annotations

import pytest

from src.web_app import serve, warmup


@pytest.fixture
def calls(monkeypatch):
    """Record the order of warm-up and Streamlit start-up without running either."""
    seen: list = []
    monkeypatch.setattr(warmup, "start", lambda: seen.append("start"))
    monkeypatch.setattr(warmup, "wait", lambda timeout=None: seen.append(("wait", timeout)) or True)

    from streamlit.web import cli

    monkeypatch.setattr(cli, "main", lambda: seen.append("streamlit") or 0)
    monkeypatch.setattr("sys.argv", ["serve", "--server.port=8501"])
    return seen


def _run():
    with pytest.raises(SystemExit):
        serve.main()


def test_default_serves_at_once(calls, monkeypatch):
    monkeypatch.delenv("FINNIE_WARM_BEFORE_SERVE", raising=False)
    _run()
    assert calls == ["start", "streamlit"]


def test_flag_waits_for_warmup_before_streamlit(calls, monkeypatch):
    monkeypatch.setenv("FINNIE_WARM_BEFORE_SERVE", "1")
    monkeypatch.delenv("FINNIE_WARMUP_TIMEOUT", raising=False)
    _run()
    assert calls == ["start", ("wait", 540.0), "streamlit"]


def test_timeout_is_configurable_and_bad_values_fall_back(calls, monkeypatch):
    monkeypatch.setenv("FINNIE_WARM_BEFORE_SERVE", "true")
    monkeypatch.setenv("FINNIE_WARMUP_TIMEOUT", "30")
    _run()
    monkeypatch.setenv("FINNIE_WARMUP_TIMEOUT", "soon")
    _run()
    assert [c for c in calls if isinstance(c, tuple)] == [("wait", 30.0), ("wait", 540.0)]


def test_serves_even_if_warmup_times_out(calls, monkeypatch):
    monkeypatch.setenv("FINNIE_WARM_BEFORE_SERVE", "1")
    monkeypatch.setattr(warmup, "wait", lambda timeout=None: calls.append("wait") or False)
    _run()
    assert calls == ["start", "wait", "streamlit"]


def test_streamlit_gets_the_app_and_passthrough_args(calls, monkeypatch):
    import sys

    monkeypatch.delenv("FINNIE_WARM_BEFORE_SERVE", raising=False)
    _run()
    assert sys.argv[:2] == ["streamlit", "run"]
    assert sys.argv[2].endswith("app.py")
    assert sys.argv[3:] == ["--server.port=8501"]
