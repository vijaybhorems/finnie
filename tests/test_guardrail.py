"""Unit tests for the pre-router guardrail node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.core.state import AgentType, FinancialData, FinnieState, UserProfile


def _make_state(query: str) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)] if query else [],
        user_profile=UserProfile(),
        financial_data=FinancialData(),
    )


class _VerdictLLM:
    """Mock LLM returning a fixed guardrail JSON verdict."""

    def __init__(self, on_topic: bool, reason: str = "test"):
        self._payload = f'{{"on_topic": {str(on_topic).lower()}, "reason": "{reason}"}}'

    def invoke(self, _messages):
        return MagicMock(content=self._payload)


class TestGuardrailBlocklist:
    """Fast-path blocklist rejects obvious NSFW/disallowed content without an LLM call."""

    def test_nsfw_term_rejected(self):
        from src.workflow.guardrail import guardrail_node

        # get_llm must NOT be called on the fast path.
        with patch("src.workflow.guardrail.get_llm") as mock_llm:
            result = guardrail_node(_make_state("show me some porn"))
            mock_llm.assert_not_called()

        assert result["is_on_topic"] is False
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE
        assert result["final_response"]

    def test_finance_word_not_falsely_blocked(self):
        """A finance query containing no blocklist terms reaches the LLM classifier."""
        from src.workflow.guardrail import guardrail_node

        with patch("src.workflow.guardrail.get_llm", return_value=_VerdictLLM(True)):
            result = guardrail_node(_make_state("What is a P/E ratio?"))

        assert result["is_on_topic"] is True


class TestGuardrailClassifier:
    """LLM classifier path: allow on-topic, reject off-topic."""

    def test_on_topic_allowed(self):
        from src.workflow.guardrail import guardrail_node

        with patch("src.workflow.guardrail.get_llm", return_value=_VerdictLLM(True, "finance")):
            result = guardrail_node(_make_state("How do bonds work?"))

        assert result["is_on_topic"] is True

    def test_off_topic_rejected(self):
        from src.workflow.guardrail import guardrail_node

        with patch("src.workflow.guardrail.get_llm", return_value=_VerdictLLM(False, "cooking")):
            result = guardrail_node(_make_state("Give me a lasagna recipe"))

        assert result["is_on_topic"] is False
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE


class TestGuardrailFailClosed:
    """Any classifier error or malformed output must reject (fail closed)."""

    def test_llm_exception_rejects(self):
        from src.workflow.guardrail import guardrail_node

        bad_llm = MagicMock()
        bad_llm.invoke.side_effect = RuntimeError("API down")
        with patch("src.workflow.guardrail.get_llm", return_value=bad_llm):
            result = guardrail_node(_make_state("Some ambiguous query"))

        assert result["is_on_topic"] is False
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE

    def test_malformed_verdict_rejects(self):
        from src.workflow.guardrail import guardrail_node

        garbage = MagicMock()
        garbage.invoke.return_value = MagicMock(content="not json")
        with patch("src.workflow.guardrail.get_llm", return_value=garbage):
            result = guardrail_node(_make_state("Some query"))

        assert result["is_on_topic"] is False


class TestGuardrailToggle:
    """Disabled guardrail passes everything through."""

    def test_disabled_passes_through(self):
        from src.workflow.guardrail import guardrail_node

        with patch("src.workflow.guardrail.get_settings") as mock_settings:
            mock_settings.return_value.guardrail.enabled = False
            result = guardrail_node(_make_state("show me some porn"))

        assert result["is_on_topic"] is True


class TestRouteAfterGuardrail:
    """The conditional edge maps the verdict to allowed/rejected."""

    def test_rejected_when_off_topic(self):
        from src.workflow.guardrail import route_after_guardrail

        state = _make_state("q")
        state.is_on_topic = False
        assert route_after_guardrail(state) == "rejected"

    def test_allowed_when_on_topic(self):
        from src.workflow.guardrail import route_after_guardrail

        state = _make_state("q")
        state.is_on_topic = True
        assert route_after_guardrail(state) == "allowed"

    def test_allowed_when_unset(self):
        from src.workflow.guardrail import route_after_guardrail

        assert route_after_guardrail(_make_state("q")) == "allowed"
