"""Tests for the LangGraph router."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.core.state import AgentType, FinnieState
from src.workflow.router import router_node


@pytest.fixture
def base_state():
    return FinnieState(messages=[HumanMessage(content="test message")])


class TestRouterNode:
    def _make_state(self, query: str) -> FinnieState:
        return FinnieState(messages=[HumanMessage(content=query)])

    @patch("src.workflow.router.get_llm")
    def test_routes_finance_qa(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"agent": "finance_qa", "reasoning": "conceptual question"}')
        mock_get_llm.return_value = mock_llm

        state = self._make_state("What is a P/E ratio?")
        result = router_node(state)

        assert result["next_agent"] == AgentType.FINANCE_QA

    @patch("src.workflow.router.get_llm")
    def test_routes_portfolio(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"agent": "portfolio", "reasoning": "portfolio analysis request"}')
        mock_get_llm.return_value = mock_llm

        state = self._make_state("Analyze my portfolio: AAPL 10 @ 150")
        result = router_node(state)

        assert result["next_agent"] == AgentType.PORTFOLIO

    @patch("src.workflow.router.get_llm")
    def test_routes_tax_education(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"agent": "tax_education", "reasoning": "tax question"}')
        mock_get_llm.return_value = mock_llm

        state = self._make_state("How does a Roth IRA work?")
        result = router_node(state)

        assert result["next_agent"] == AgentType.TAX_EDUCATION

    @patch("src.workflow.router.get_llm")
    def test_fallback_on_json_parse_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Sorry I cannot route this")
        mock_get_llm.return_value = mock_llm

        state = self._make_state("some question")
        result = router_node(state)

        # Should default to finance_qa
        assert result["next_agent"] == AgentType.FINANCE_QA

    @patch("src.workflow.router.get_llm")
    def test_fallback_on_exception(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")
        mock_get_llm.return_value = mock_llm

        state = self._make_state("some question")
        result = router_node(state)

        assert result["next_agent"] == AgentType.FINANCE_QA
        assert "error" in result["router_reasoning"].lower()

    def test_empty_messages_defaults_to_finance_qa(self):
        state = FinnieState(messages=[])
        with patch("src.workflow.router.get_llm") as mock_get_llm:
            result = router_node(state)
        assert result["next_agent"] == AgentType.FINANCE_QA
