"""End-to-end workflow evals for run_workflow.

Validates the full compiled graph (guardrail → router → agent → END), including
the guardrail short-circuit path, using offline stubs.

Run with:
    pytest tests/evals/test_workflow_e2e_evals.py -v
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _clear_agent_singletons():
    """Reset the lazily-built agent registry so patched deps take effect."""
    import src.workflow.graph as graph_mod
    graph_mod._AGENTS.clear()  # noqa: SLF001 — test-only reset


def _verdict_llm(payload: dict):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(payload))
    return llm


class TestGuardrailShortCircuit:
    """Off-topic / NSFW queries never reach an agent."""

    def test_nsfw_blocked_offline(self):
        """Blocklist fast-path needs no LLM — fully offline."""
        from src.workflow.graph import run_workflow
        result = run_workflow("show me some porn")
        assert result["agent_used"] == "out_of_scope"
        assert result["final_response"]

    def test_off_topic_blocked_via_classifier(self):
        with patch(
            "src.workflow.guardrail.get_llm",
            return_value=_verdict_llm({"on_topic": False, "reason": "cooking"}),
        ):
            from src.workflow.graph import run_workflow
            result = run_workflow("Give me a lasagna recipe")
        assert result["agent_used"] == "out_of_scope"


class TestInScopeRouting:
    """In-scope finance queries pass the guardrail and reach the right agent."""

    def test_routes_to_finance_qa(self, retriever):
        _clear_agent_singletons()
        with patch(
            "src.workflow.guardrail.get_llm",
            return_value=_verdict_llm({"on_topic": True, "reason": "finance"}),
        ), patch(
            "src.workflow.router.get_llm",
            return_value=_verdict_llm({"agent": "finance_qa", "reasoning": "concept"}),
        ), patch("src.agents.base_agent.get_llm"), patch.object(
            __import__("src.agents.base_agent", fromlist=["BaseAgent"]).BaseAgent,
            "_invoke_llm",
            lambda self, state, additional_system="": "Educational explanation.",
        ), patch(
            "src.agents.finance_qa_agent.get_retriever", return_value=retriever
        ), patch("src.agents.finance_qa_agent.FredClient") as mock_fred:
            mock_fred.return_value.get_macro_snapshot.return_value = {}
            from src.workflow.graph import run_workflow
            result = run_workflow("What is a P/E ratio?")

        assert result["agent_used"] == "finance_qa"
        assert "not financial advice" in result["final_response"].lower()

    def test_result_shape(self):
        """run_workflow always returns the documented keys."""
        from src.workflow.graph import run_workflow
        result = run_workflow("show me some porn")  # offline blocklist path
        for key in ("final_response", "agent_used", "router_reasoning",
                    "financial_data", "rag_context", "messages"):
            assert key in result
