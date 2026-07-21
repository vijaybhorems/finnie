"""Regression tests for list-shaped LLM content normalization.

Anthropic models (e.g. Claude Sonnet 5) can return ``response.content`` as a list
of content blocks (``[{"type": "text", "text": "..."}]``) instead of a plain
string. Code that did ``content + disclaimer`` or ``content.strip()`` crashed on
the list form. ``message_text`` normalizes both shapes; these tests guard every
model-output read site (agents via ``_invoke_llm``, router, guardrail) against a
future model change silently reintroducing the bug.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.core.llm import message_text
from src.core.state import AgentType, FinancialData, FinnieState, UserProfile


def _make_state(query: str) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(),
        financial_data=FinancialData(),
    )


class TestMessageText:
    """Unit tests for the normalizer itself."""

    def test_plain_string_passthrough(self):
        assert message_text(SimpleNamespace(content="hello")) == "hello"

    def test_list_of_text_blocks_joined(self):
        resp = SimpleNamespace(content=[
            {"type": "text", "text": "A P/E ratio is "},
            {"type": "text", "text": "price over earnings."},
        ])
        assert message_text(resp) == "A P/E ratio is price over earnings."

    def test_non_text_blocks_dropped(self):
        resp = SimpleNamespace(content=[
            {"type": "thinking", "thinking": "internal reasoning"},
            {"type": "text", "text": "Answer."},
            {"type": "tool_use", "name": "search", "input": {}},
        ])
        assert message_text(resp) == "Answer."

    def test_object_style_blocks(self):
        resp = SimpleNamespace(content=[SimpleNamespace(text="chunk1"), SimpleNamespace(text="chunk2")])
        assert message_text(resp) == "chunk1chunk2"

    def test_empty_list_returns_empty_string(self):
        assert message_text(SimpleNamespace(content=[])) == ""

    def test_bare_string_input(self):
        # Accepts a raw value too, not just an object with .content
        assert message_text("just text") == "just text"


class TestAgentListContent:
    """The reported crash: list content flowing into ``_add_disclaimer``."""

    @patch("src.agents.finance_qa_agent.get_retriever")
    @patch("src.agents.base_agent.get_llm")
    def test_finance_qa_agent_handles_list_content(self, mock_get_llm, mock_retriever):
        # LLM returns list-shaped content (Sonnet 5 form) — must not raise
        # "can only concatenate list (not str) to list". Agents call the LLM via a
        # ``prompt | llm`` chain, which coerces the mock to a RunnableLambda and
        # *calls* it, so set the call (``.return_value``) not ``.invoke``.
        resp = MagicMock(content=[{"type": "text", "text": "A P/E ratio is price over earnings."}])
        mock_get_llm.return_value.return_value = resp
        mock_get_llm.return_value.invoke.return_value = resp
        mock_retriever.return_value = MagicMock(get_context=MagicMock(return_value=""))

        with patch("src.agents.finance_qa_agent.FredClient") as mock_fred:
            mock_fred.return_value.get_macro_snapshot.return_value = {}

            from src.agents.finance_qa_agent import FinanceQAAgent
            result = FinanceQAAgent().run(_make_state("What is a P/E ratio?"))

        assert result["final_response"].startswith("A P/E ratio is price over earnings.")
        assert "not financial advice" in result["final_response"].lower()


class TestRouterListContent:
    @patch("src.workflow.router.get_llm")
    def test_router_parses_list_content(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=[{"type": "text", "text": '{"agent": "finance_qa", "reasoning": "conceptual"}'}]
        )
        mock_get_llm.return_value = mock_llm

        from src.workflow.router import router_node
        result = router_node(_make_state("What is a P/E ratio?"))
        assert result["next_agent"] == AgentType.FINANCE_QA


class TestGuardrailListContent:
    @patch("src.workflow.guardrail.get_llm")
    def test_guardrail_parses_list_content(self, mock_get_llm):
        # A finance query with no blocklist term reaches the LLM classifier;
        # list-shaped verdict content must parse, not stringify into a rejection.
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=[{"type": "text", "text": '{"on_topic": true, "reason": "finance"}'}]
        )
        mock_get_llm.return_value = mock_llm

        from src.workflow.guardrail import guardrail_node
        result = guardrail_node(_make_state("How do bonds work?"))
        assert result["is_on_topic"] is True
