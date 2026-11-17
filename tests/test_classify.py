"""Tests for the merged guardrail + router classify node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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


class TestVerdictCache:
    """Early classification (voice, while the user speaks) is reused by the graph."""

    _MARKET = Verdict(on_topic=True, agent="market_analysis", needs_macro=False, reason="price")

    def test_same_question_classified_once(self):
        from src.workflow.classify import classify_query

        llm = _mock_llm(self._MARKET)
        with patch("src.workflow.classify.get_llm", return_value=llm):
            classify_query("How is Apple stock doing today")  # early, from the transcript
            result = classify_node(_make_state("How is Apple stock doing today?"))
        assert result["next_agent"] == AgentType.MARKET_ANALYSIS
        assert llm.with_structured_output.return_value.invoke.call_count == 1

    def test_different_question_is_classified(self):
        from src.workflow.classify import classify_query

        llm = _mock_llm(self._MARKET)
        with patch("src.workflow.classify.get_llm", return_value=llm):
            classify_query("How is Apple")  # partial transcript
            classify_node(_make_state("How is Apple stock doing today?"))
        assert llm.with_structured_output.return_value.invoke.call_count == 2

    def test_errors_are_not_cached(self):
        llm = _mock_llm(self._MARKET)
        llm.with_structured_output.return_value.invoke.side_effect = [RuntimeError("boom"), self._MARKET]
        with patch("src.workflow.classify.get_llm", return_value=llm):
            first = classify_node(_make_state("How is Apple stock doing today?"))
            second = classify_node(_make_state("How is Apple stock doing today?"))
        assert first["next_agent"] == AgentType.OUT_OF_SCOPE  # still fails closed
        assert second["next_agent"] == AgentType.MARKET_ANALYSIS

    def test_blocklist_still_wins_over_a_cached_verdict(self):
        from src.workflow.classify import classify_query

        on_topic = Verdict(on_topic=True, agent="finance_qa", needs_macro=False, reason="x")
        with patch("src.workflow.classify.get_llm", return_value=_mock_llm(on_topic)):
            classify_query("show me some porn")
            result = classify_node(_make_state("show me some porn"))
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE


class TestTickers:
    """The classifier names tickers for companies the user names in words."""

    def test_clean_tickers_normalises_and_filters(self):
        from src.workflow.classify import _clean_tickers

        assert _clean_tickers(["aapl", "$NVDA", "BRK.B", "AAPL", "not a ticker", "", "TOOLONG"]) == [
            "AAPL", "NVDA", "BRK-B",
        ]

    def test_clean_tickers_caps_the_list(self):
        from src.workflow.classify import _MAX_TICKERS, _clean_tickers

        assert len(_clean_tickers(["A", "B", "C", "D", "E", "F", "G"])) == _MAX_TICKERS

    def test_node_returns_cleaned_tickers(self):
        verdict = Verdict(on_topic=True, agent="market_analysis", needs_macro=False,
                          reason="stock", tickers=["aapl", "BRK.B"])
        with patch("src.workflow.classify.get_llm", return_value=_mock_llm(verdict)):
            result = classify_node(_make_state("How are Apple and Berkshire doing?"))

        assert result["query_tickers"] == ["AAPL", "BRK-B"]

    def test_tickers_default_to_empty(self):
        verdict = Verdict(on_topic=True, agent="finance_qa", needs_macro=False, reason="concept")
        with patch("src.workflow.classify.get_llm", return_value=_mock_llm(verdict)):
            assert classify_node(_make_state("What is a P/E ratio?"))["query_tickers"] == []


class TestConversationContext:
    """The previous turn, so follow-ups like "What's the exact dollar amount?" make sense."""

    _PREV = ("If I invest $10,000 at 7% for 10 years, how much will I have?",
             "At 7% a year, $10,000 grows to roughly $19,700 after ten years.")
    _GOALS = Verdict(on_topic=True, agent="goal_planning", follow_up=True, reason="follow-up on growth")

    def _follow_up(self, query: str = "What's the exact dollar amount?") -> FinnieState:
        state = _make_state(query)
        state.messages = [HumanMessage(content=self._PREV[0]), AIMessage(content=self._PREV[1]), *state.messages]
        return state

    def test_previous_turn_on_a_first_turn_is_empty(self):
        from src.workflow.classify import previous_turn

        assert previous_turn(_make_state("What is a bond?").messages) == ("", "")

    def test_previous_turn_takes_the_last_exchange(self):
        from src.workflow.classify import previous_turn

        messages = [HumanMessage(content="old"), AIMessage(content="old answer"), *self._follow_up().messages]
        assert previous_turn(messages) == self._PREV

    def test_unanswered_previous_question_has_no_answer(self):
        from src.workflow.classify import previous_turn

        messages = [HumanMessage(content="What is a bond?"), HumanMessage(content="and a stock?")]
        assert previous_turn(messages) == ("What is a bond?", "")

    def test_context_is_empty_without_a_previous_question(self):
        from src.workflow.classify import conversation_context

        assert conversation_context() == ""
        assert conversation_context("", "an answer") == ""

    def test_context_clips_long_answers(self):
        from src.workflow.classify import _CONTEXT_ANSWER_CHARS, conversation_context

        context = conversation_context("What is a bond?", "word " * 500)
        answer = context.splitlines()[-1]
        assert answer.endswith("...")
        assert len(answer) < _CONTEXT_ANSWER_CHARS + 40

    def test_follow_up_goes_to_the_conversations_agent(self):
        llm = _mock_llm(self._GOALS)
        with patch("src.workflow.classify.get_llm", return_value=llm):
            result = classify_node(self._follow_up())

        assert result["next_agent"] == AgentType.GOAL_PLANNING
        assert result["follow_up"] is True
        system, user = llm.with_structured_output.return_value.invoke.call_args.args[0]
        assert self._PREV[0] in user["content"] and "19,700" in user["content"]
        assert self._PREV[0] not in system["content"]  # per request, never the system prompt
        assert user["content"].endswith("Classify this query: What's the exact dollar amount?")

    def test_first_turn_sends_the_query_alone(self):
        llm = _mock_llm(self._GOALS)
        with patch("src.workflow.classify.get_llm", return_value=llm):
            result = classify_node(_make_state("What is a bond?"))

        _, user = llm.with_structured_output.return_value.invoke.call_args.args[0]
        assert user["content"] == "Classify this query: What is a bond?"
        assert result["follow_up"] is False  # nothing to follow, whatever the model said

    def test_same_words_after_a_different_turn_are_classified_again(self):
        from src.workflow.classify import classify_query, conversation_context

        llm = _mock_llm(self._GOALS)
        with patch("src.workflow.classify.get_llm", return_value=llm):
            classify_query("What about the income limits?", context=conversation_context("How does a Roth IRA work?"))
            classify_query("What about the income limits?", context=conversation_context("Can I deduct mortgage interest?"))
            classify_query("What about the income limits?", context=conversation_context("How does a Roth IRA work?"))
        assert llm.with_structured_output.return_value.invoke.call_count == 2

    def test_blocklist_still_wins_with_context(self):
        with patch("src.workflow.classify.get_llm") as mock_llm:
            result = classify_node(self._follow_up("now show me porn"))
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE
        mock_llm.assert_not_called()

    def test_error_with_context_fails_closed(self):
        llm = _mock_llm(self._GOALS)
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("boom")
        with patch("src.workflow.classify.get_llm", return_value=llm):
            result = classify_node(self._follow_up())
        assert result["next_agent"] == AgentType.OUT_OF_SCOPE


class TestAcknowledgements:
    """A bare "Stop." / "wait" / "never mind" gets a short reply: no refusal, no agent."""

    @pytest.mark.parametrize("query", [
        "Stop.", "stop", "Wait!", "wait a second", "Never mind.", "nevermind", "Hold on",
        "okay stop", "no no stop", "um never mind", "Forget it", "stop please", "Finnie, stop.",
        "hang on", "That's enough.", "Cancel that",
    ])
    def test_acknowledged_without_a_model_call(self, query):
        from src.workflow.classify import ACKNOWLEDGEMENT

        with patch("src.workflow.classify.get_llm") as mock_llm:
            result = classify_node(_make_state(query))
        mock_llm.assert_not_called()
        assert result["is_on_topic"] is True
        assert result["next_agent"] == AgentType.ACKNOWLEDGE
        assert result["final_response"] == ACKNOWLEDGEMENT
        assert result["messages"][0].content == ACKNOWLEDGEMENT

    @pytest.mark.parametrize("query", [
        "Wait, what's a Roth IRA?", "What is a stop-loss order?", "stop loss", "Should I stop investing?",
        "never mind that, how do bonds work?", "okay", "no",
    ])
    def test_questions_are_classified(self, query):
        from src.workflow.classify import is_acknowledgement

        assert not is_acknowledgement(query)

    def test_route_after_classify_ends_the_turn(self):
        state = _make_state("Stop.")
        state.is_on_topic, state.next_agent = True, AgentType.ACKNOWLEDGE
        assert route_after_classify(state) == "acknowledged"


class TestReasonSetting:
    """llm.classifier_reason: off on voice, where the written reason only costs time."""

    def test_reason_asked_for_by_default(self):
        from src.workflow.classify import classify_query

        llm = _mock_llm(Verdict(on_topic=True, agent="finance_qa", reason="concept"))
        with patch("src.workflow.classify.get_llm", return_value=llm):
            verdict = classify_query("What is a bond?")
        llm.with_structured_output.assert_called_once_with(Verdict)
        assert verdict.reason == "concept"

    def test_without_reason_the_decision_alone_is_asked_for(self):
        from src.core.config import get_settings
        from src.workflow.classify import Routing, classify_query

        get_settings().llm.classifier_reason = False
        llm = _mock_llm(Routing(on_topic=True, agent="tax_education", tickers=[]))
        with patch("src.workflow.classify.get_llm", return_value=llm):
            verdict = classify_query("What is a Roth IRA?")
            result = classify_node(_make_state("What is a Roth IRA?"))  # the same verdict, reused
        llm.with_structured_output.assert_called_once_with(Routing)
        assert isinstance(verdict, Verdict) and verdict.reason == "" and verdict.agent == "tax_education"
        assert result["next_agent"] == AgentType.TAX_EDUCATION

    def test_a_verdict_without_reason_is_not_reused_with_one(self):
        """The setting is part of the cache key: a process that wants reasons gets them."""
        from src.core.config import get_settings
        from src.workflow.classify import Routing, classify_query

        settings = get_settings()
        settings.llm.classifier_reason = False
        with patch("src.workflow.classify.get_llm", return_value=_mock_llm(Routing(on_topic=True))):
            classify_query("What is a bond?")
        settings.llm.classifier_reason = True
        llm = _mock_llm(Verdict(on_topic=True, reason="concept"))
        with patch("src.workflow.classify.get_llm", return_value=llm):
            assert classify_query("What is a bond?").reason == "concept"

    def test_the_voice_process_turns_it_off(self, monkeypatch):
        from src.core.config import get_settings
        from src.voice.runtime import configure_process

        monkeypatch.delenv("FINNIE_CLASSIFIER_MODEL", raising=False)
        configure_process()
        assert get_settings().llm.classifier_reason is False
