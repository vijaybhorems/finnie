"""Tests for the merged guardrail + router classify node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.core.state import AgentType, FinancialData, FinnieState, UserProfile
from src.workflow.classify import Verdict, classify_node, route_after_classify


def _make_state(query: str) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)] if query else [],
        user_profile=UserProfile(),
        financial_data=FinancialData(),
    )


def _mock_llm(verdict: Verdict) -> MagicMock:
    structured = MagicMock()
    structured.invoke.return_value = verdict
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


class TestBlocklistFastPath:
    def test_nsfw_rejected_without_llm_call(self):
        with patch("src.workflow.classify.get_llm") as mock_llm:
            result = classify_node(_make_state("show me some porn"))
            mock_llm.assert_not_called()

        assert result["is_on_topic"] is False
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE
        assert result["final_response"]


class TestRouting:
    def test_routes_to_named_agent(self):
        verdict = Verdict(on_topic=True, agent="portfolio", needs_macro=False, reason="holdings")
        with patch("src.workflow.classify.get_llm", return_value=_mock_llm(verdict)):
            result = classify_node(_make_state("Analyze my portfolio: AAPL 10 @ 150"))

        assert result["is_on_topic"] is True
        assert result["next_agent"] == AgentType.PORTFOLIO
        assert result["needs_macro"] is False

    def test_propagates_needs_macro(self):
        verdict = Verdict(on_topic=True, agent="finance_qa", needs_macro=True, reason="rates")
        with patch("src.workflow.classify.get_llm", return_value=_mock_llm(verdict)):
            result = classify_node(_make_state("How do current rates affect bonds?"))

        assert result["needs_macro"] is True

    def test_dict_verdict_is_coerced(self):
        """with_structured_output can hand back a dict on some model paths."""
        structured = MagicMock()
        structured.invoke.return_value = {
            "on_topic": True,
            "agent": "tax_education",
            "needs_macro": False,
            "reason": "roth",
        }
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        with patch("src.workflow.classify.get_llm", return_value=llm):
            result = classify_node(_make_state("How does a Roth IRA work?"))

        assert result["next_agent"] == AgentType.TAX_EDUCATION

    def test_empty_query_defaults_to_finance_qa(self):
        with patch("src.workflow.classify.get_llm") as mock_llm:
            result = classify_node(_make_state(""))
            mock_llm.assert_not_called()

        assert result["next_agent"] == AgentType.FINANCE_QA
        assert result["is_on_topic"] is True


class TestFailClosed:
    def test_off_topic_rejected(self):
        verdict = Verdict(on_topic=False, agent="finance_qa", reason="cooking")
        with patch("src.workflow.classify.get_llm", return_value=_mock_llm(verdict)):
            result = classify_node(_make_state("Give me a lasagna recipe"))

        assert result["is_on_topic"] is False
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE

    def test_llm_exception_rejects(self):
        llm = MagicMock()
        llm.with_structured_output.side_effect = RuntimeError("API down")
        with patch("src.workflow.classify.get_llm", return_value=llm):
            result = classify_node(_make_state("Some ambiguous query"))

        assert result["is_on_topic"] is False
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE

    def test_none_verdict_rejects(self):
        with patch("src.workflow.classify.get_llm", return_value=_mock_llm(None)):
            result = classify_node(_make_state("Some query"))

        assert result["is_on_topic"] is False

    def test_disabled_guardrail_still_routes_on_error(self):
        """With the scope gate off there is nothing to fail closed to."""
        llm = MagicMock()
        llm.with_structured_output.side_effect = RuntimeError("API down")
        with patch("src.workflow.classify.get_llm", return_value=llm), patch(
            "src.workflow.classify.get_settings"
        ) as mock_settings:
            mock_settings.return_value.guardrail.enabled = False
            result = classify_node(_make_state("show me some porn"))

        assert result["is_on_topic"] is True
        assert result["next_agent"] == AgentType.FINANCE_QA


class TestRouteAfterClassify:
    def test_rejected_when_off_topic(self):
        state = _make_state("q")
        state.is_on_topic = False
        assert route_after_classify(state) == "rejected"

    def test_allowed_when_on_topic(self):
        state = _make_state("q")
        state.is_on_topic = True
        assert route_after_classify(state) == "allowed"
