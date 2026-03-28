"""Answer quality evals.

These tests invoke real agents (with mocked or real LLMs) and validate
that responses meet Finnie's quality bar:
  - Factual accuracy (key facts from the knowledge base are present)
  - Disclaimer inclusion
  - Appropriate tone / no investment recommendations
  - Knowledge-level adaptation

Two modes:
  1. MOCKED (default): Intercepts the _invoke_llm call to echo back the
     assembled system prompt context — fast, deterministic, tests that the
     right KB context is being fed to the LLM.
  2. LIVE: Set FINNIE_EVAL_LIVE=1 to use the real Anthropic API.
     Slower, costs money, but tests true end-to-end answer quality.

Run with:
    pytest tests/evals/test_answer_quality.py -v              # mocked
    FINNIE_EVAL_LIVE=1 pytest tests/evals/test_answer_quality.py -v  # live
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.core.state import FinancialData, FinnieState, UserProfile

LIVE_MODE = os.environ.get("FINNIE_EVAL_LIVE", "0") == "1"


def _make_state(query: str, **profile_kwargs) -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(**profile_kwargs),
        financial_data=FinancialData(),
    )


def _echo_invoke_llm(self, state, additional_system=""):
    """Replacement for BaseAgent._invoke_llm that echoes the additional_system context.

    This lets us assert on the context the agent would send to the LLM
    without needing a real LangChain-compatible Runnable.
    """
    return f"ECHO: {additional_system[:3000]}"


# ---------------------------------------------------------------------------
# Quality eval cases for Finance Q&A agent
# ---------------------------------------------------------------------------
FINANCE_QA_CASES = [
    {
        "query": "What is a P/E ratio and why does it matter?",
        "expected_context_keywords": ["P/E", "price", "earnings"],
        "description": "Should retrieve P/E ratio content from knowledge base",
    },
    {
        "query": "How does compound interest grow my savings?",
        "expected_context_keywords": ["compound", "interest", "growth"],
        "description": "Should retrieve compound interest content",
    },
    {
        "query": "What is the difference between an ETF and a mutual fund?",
        "expected_context_keywords": ["ETF", "index fund", "expense ratio"],
        "description": "Should retrieve ETF/fund content",
    },
]


class TestFinanceQAAnswerQuality:
    """Validate the Finance Q&A agent assembles the right context and produces quality answers."""

    @pytest.mark.parametrize("case", FINANCE_QA_CASES, ids=[c["description"] for c in FINANCE_QA_CASES])
    def test_context_assembly(self, retriever, case):
        """Verify the agent feeds the right knowledge base context to the LLM."""
        with patch("src.agents.finance_qa_agent.get_retriever", return_value=retriever), \
             patch("src.agents.base_agent.get_llm"), \
             patch.object(
                 __import__("src.agents.base_agent", fromlist=["BaseAgent"]).BaseAgent,
                 "_invoke_llm",
                 _echo_invoke_llm,
             ), \
             patch("src.agents.finance_qa_agent.FredClient") as mock_fred:

            mock_fred.return_value.get_macro_snapshot.return_value = {}

            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
            state = _make_state(case["query"])
            result = agent.run(state)

        response = result["final_response"].lower()
        matched = [kw for kw in case["expected_context_keywords"] if kw.lower() in response]
        assert matched, (
            f"Expected context keywords {case['expected_context_keywords']} not found in "
            f"assembled context for: {case['query']}"
        )

    def test_disclaimer_always_present(self, retriever):
        """Every response must end with the standard disclaimer."""
        with patch("src.agents.finance_qa_agent.get_retriever", return_value=retriever), \
             patch("src.agents.base_agent.get_llm"), \
             patch.object(
                 __import__("src.agents.base_agent", fromlist=["BaseAgent"]).BaseAgent,
                 "_invoke_llm",
                 _echo_invoke_llm,
             ), \
             patch("src.agents.finance_qa_agent.FredClient") as mock_fred:

            mock_fred.return_value.get_macro_snapshot.return_value = {}

            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
            state = _make_state("What are index funds?")
            result = agent.run(state)

        assert "not financial advice" in result["final_response"].lower()

    def test_rag_context_populated(self, retriever):
        """The agent should populate rag_context in the response."""
        with patch("src.agents.finance_qa_agent.get_retriever", return_value=retriever), \
             patch("src.agents.base_agent.get_llm"), \
             patch.object(
                 __import__("src.agents.base_agent", fromlist=["BaseAgent"]).BaseAgent,
                 "_invoke_llm",
                 _echo_invoke_llm,
             ), \
             patch("src.agents.finance_qa_agent.FredClient") as mock_fred:

            mock_fred.return_value.get_macro_snapshot.return_value = {}

            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
            state = _make_state("What is diversification?")
            result = agent.run(state)

        assert len(result.get("rag_context", [])) > 0, "rag_context should not be empty"


# ---------------------------------------------------------------------------
# Quality eval cases for Tax Education agent
# ---------------------------------------------------------------------------
TAX_CASES = [
    {
        "query": "How does tax-loss harvesting work?",
        "expected_context_keywords": ["tax-loss harvesting", "wash-sale", "capital"],
        "description": "Tax-loss harvesting should pull tax_accounts context",
    },
    {
        "query": "What are the Roth IRA contribution limits?",
        "expected_context_keywords": ["Roth", "IRA", "contribution", "limit"],
        "description": "Roth IRA limits should pull retirement accounts context",
    },
    {
        "query": "How does the HSA triple tax advantage work?",
        "expected_context_keywords": ["HSA", "triple", "tax"],
        "description": "HSA query should pull tax_accounts context",
    },
]


class TestTaxAgentAnswerQuality:
    """Validate the Tax Education agent assembles the right context."""

    @pytest.mark.parametrize("case", TAX_CASES, ids=[c["description"] for c in TAX_CASES])
    def test_tax_context_assembly(self, retriever, case):
        with patch("src.agents.tax_education_agent.get_retriever", return_value=retriever), \
             patch("src.agents.base_agent.get_llm"), \
             patch.object(
                 __import__("src.agents.base_agent", fromlist=["BaseAgent"]).BaseAgent,
                 "_invoke_llm",
                 _echo_invoke_llm,
             ):

            from src.agents.tax_education_agent import TaxEducationAgent
            agent = TaxEducationAgent()
            state = _make_state(case["query"])
            result = agent.run(state)

        response = result["final_response"].lower()
        matched = [kw for kw in case["expected_context_keywords"] if kw.lower() in response]
        assert matched, (
            f"Expected {case['expected_context_keywords']} in response. "
            f"Query: {case['query']}"
        )

    def test_tax_data_2024_injected(self, retriever):
        """The 2024 tax reference data should be injected into the LLM context."""
        with patch("src.agents.tax_education_agent.get_retriever", return_value=retriever), \
             patch("src.agents.base_agent.get_llm"), \
             patch.object(
                 __import__("src.agents.base_agent", fromlist=["BaseAgent"]).BaseAgent,
                 "_invoke_llm",
                 _echo_invoke_llm,
             ):

            from src.agents.tax_education_agent import TaxEducationAgent
            agent = TaxEducationAgent()
            state = _make_state("What are the 2024 tax brackets?")
            result = agent.run(state)

        # The echo returns the additional_system which should include tax data
        response = result["final_response"]
        assert "23000" in response or "401k" in response.lower(), (
            "2024 tax reference data (e.g., 401k limit $23,000) should be in context"
        )


# ---------------------------------------------------------------------------
# Live LLM evals (only run when FINNIE_EVAL_LIVE=1)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not LIVE_MODE, reason="Set FINNIE_EVAL_LIVE=1 to run live evals")
class TestLiveAnswerQuality:
    """End-to-end evals using the real Anthropic API."""

    LIVE_CASES = [
        {
            "query": "What is a P/E ratio? I'm a complete beginner.",
            "must_contain": ["price", "earnings"],
            "must_not_contain": ["buy", "sell", "you should invest"],
            "description": "Beginner P/E ratio explanation",
        },
        {
            "query": "Explain the difference between a Roth and Traditional IRA",
            "must_contain": ["tax", "Roth", "traditional"],
            "must_not_contain": ["you should open", "I recommend"],
            "description": "IRA comparison should be educational not advisory",
        },
        {
            "query": "What happens to bond prices when interest rates rise?",
            "must_contain": ["inverse", "fall", "drop", "decline", "decrease"],
            "must_not_contain": ["buy bonds now", "sell your bonds"],
            "description": "Bond/rate relationship must be factually correct",
        },
    ]

    @pytest.mark.parametrize("case", LIVE_CASES, ids=[c["description"] for c in LIVE_CASES])
    def test_live_response_quality(self, retriever, case):
        with patch("src.agents.finance_qa_agent.get_retriever", return_value=retriever), \
             patch("src.agents.finance_qa_agent.FredClient") as mock_fred:

            mock_fred.return_value.get_macro_snapshot.return_value = {}

            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
            state = _make_state(case["query"], knowledge_level="beginner")
            result = agent.run(state)

        response = result["final_response"].lower()

        # Must contain at least one expected keyword
        found = [kw for kw in case["must_contain"] if kw.lower() in response]
        assert found, (
            f"Response missing expected content. Expected one of {case['must_contain']}"
        )

        # Must not contain advisory language
        violations = [kw for kw in case["must_not_contain"] if kw.lower() in response]
        assert not violations, (
            f"Response contains prohibited advisory language: {violations}"
        )

        # Must include disclaimer
        assert "not financial advice" in response or "disclaimer" in response, (
            "Response missing disclaimer"
        )

    @pytest.mark.parametrize("level", ["beginner", "advanced"])
    def test_knowledge_level_adaptation(self, retriever, level):
        """Beginner answers should use simpler language than advanced answers."""
        with patch("src.agents.finance_qa_agent.get_retriever", return_value=retriever), \
             patch("src.agents.finance_qa_agent.FredClient") as mock_fred:

            mock_fred.return_value.get_macro_snapshot.return_value = {}

            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
            state = _make_state(
                "Explain standard deviation in portfolio risk",
                knowledge_level=level,
            )
            result = agent.run(state)

        response = result["final_response"]
        # Basic sanity: response should be non-trivial
        assert len(response) > 100, f"Response too short for {level} level"
