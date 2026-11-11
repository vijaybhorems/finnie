"""API keys never reach logs or the error values agents put in prompts."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
import structlog
from structlog.testing import capture_logs

from src.utils.redact import MASK, describe_request_error, make_redact_processor, redact

FRED_KEY = "fredkey0123456789abcdef"
AV_KEY = "avkey0123456789"
NEWS_KEY = "newskey0123456789"


def _http_error(url: str, status: int = 400, reason: str = "Bad Request") -> requests.HTTPError:
    """The error requests raises: its message carries the full URL."""
    resp = requests.Response()
    resp.status_code, resp.reason, resp.url = status, reason, url
    return requests.HTTPError(f"{status} Client Error: {reason} for url: {url}", response=resp)


def _no_cache() -> MagicMock:
    cache = MagicMock()
    cache.get.return_value = None
    cache.get_stale.return_value = None
    cache.cache_key.return_value = "k"
    return cache


class TestRedact:
    def test_masks_known_secret(self):
        assert redact(f"failed with {FRED_KEY} inside", [FRED_KEY]) == f"failed with {MASK} inside"

    @pytest.mark.parametrize("param", ["api_key", "apikey", "apiKey", "api-key", "token", "access_token"])
    def test_masks_secret_query_values_even_when_unknown(self, param):
        out = redact(f"for url: https://x.test/q?a=1&{param}=s3cr3tvalue&b=2")
        assert "s3cr3tvalue" not in out and f"{param}={MASK}&b=2" in out

    def test_short_values_are_not_treated_as_secrets(self):
        # LiveKit's dev default is "secret"; masking it would mangle ordinary words.
        assert redact("a secret plan", ["secret"]) == "a secret plan"

    def test_processor_cleans_nested_values(self):
        clean = make_redact_processor([FRED_KEY])
        event = clean(None, "error", {"event": "x", "error": f"k={FRED_KEY}", "params": {"u": [FRED_KEY]}})
        assert FRED_KEY not in json.dumps(event)


class TestDescribeRequestError:
    def test_http_error_drops_the_url(self):
        exc = _http_error(f"https://api.test/obs?series_id=X&api_key={FRED_KEY}")
        assert describe_request_error(exc, [FRED_KEY]) == "HTTP 400 Bad Request"

    def test_timeout_and_connection(self):
        assert describe_request_error(requests.Timeout("t")) == "request timed out"
        assert describe_request_error(requests.ConnectionError(f"x?api_key={FRED_KEY}")) == "connection error"

    def test_other_errors_are_redacted(self):
        assert FRED_KEY not in describe_request_error(ValueError(f"bad {FRED_KEY}"), [FRED_KEY])


class TestProviders:
    def test_fred_error_has_no_key_in_result_or_log(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", FRED_KEY)
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=X&api_key={FRED_KEY}&file_type=json"
        with patch("src.data.fred_client.get_cache", return_value=_no_cache()), \
                patch("src.data.fred_client.requests.get", side_effect=_http_error(url)), \
                capture_logs() as logs:
            from src.data.fred_client import FredClient

            result = FredClient().get_series_latest("X")
        assert result["error"] == "HTTP 400 Bad Request"
        assert FRED_KEY not in json.dumps(logs)
        assert any(e["event"] == "fred_request_error" for e in logs)

    def test_fred_series_are_all_fred_ids(self):
        # The old Shiller P/E entry was a Quandl id, so every snapshot made a failing call.
        from src.data.fred_client import FRED_SERIES

        assert all("/" not in sid for sid in FRED_SERIES.values())

    def test_alpha_vantage_error_has_no_key_in_result_or_log(self, monkeypatch):
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", AV_KEY)
        url = f"https://www.alphavantage.co/query?function=RSI&apikey={AV_KEY}"
        with patch("src.data.alpha_vantage_client.get_cache", return_value=_no_cache()), \
                patch("src.data.alpha_vantage_client.requests.get", side_effect=_http_error(url, 500, "Server Error")), \
                capture_logs() as logs:
            from src.data.alpha_vantage_client import AlphaVantageClient

            result = AlphaVantageClient()._call({"function": "RSI", "symbol": "AAPL"})
        assert result["error"] == "HTTP 500 Server Error"
        assert AV_KEY not in json.dumps(logs)
        logged = next(e for e in logs if e["event"] == "alpha_vantage_request_error")
        assert logged["params"] == {"function": "RSI", "symbol": "AAPL"}

    def test_newsapi_key_goes_in_a_header_not_the_url(self, monkeypatch):
        monkeypatch.setenv("NEWS_API_KEY", NEWS_KEY)
        resp = MagicMock()
        resp.json.return_value = {"articles": []}
        with patch("src.data.news_client.get_cache", return_value=_no_cache()), \
                patch("src.data.news_client.requests.get", return_value=resp) as get:
            from src.data.news_client import NewsClient

            NewsClient().get_financial_headlines("rates")
        kwargs = get.call_args.kwargs
        assert NEWS_KEY not in json.dumps(kwargs["params"])
        assert kwargs["headers"] == {"X-Api-Key": NEWS_KEY}


def test_setup_logging_masks_secrets_everywhere(monkeypatch, capsys):
    monkeypatch.setenv("FRED_API_KEY", FRED_KEY)
    monkeypatch.setenv("ELEVEN_API_KEY", "elevenkey0123456789")
    from src.utils.logger import get_logger, setup_logging

    try:
        setup_logging()
        get_logger("t").error("oops", detail=f"saw {FRED_KEY} and elevenkey0123456789")
        out = capsys.readouterr().out
    finally:
        structlog.reset_defaults()
    assert "oops" in out
    assert FRED_KEY not in out and "elevenkey0123456789" not in out
