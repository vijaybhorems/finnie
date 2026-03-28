"""Tax Education Agent — explains tax concepts and account types."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.core.state import FinnieState
from src.rag.retriever import get_retriever

# Static tax data (updated less frequently than market data)
TAX_DATA_2024 = {
    "federal_brackets_single": [
        {"rate": "10%", "income": "Up to $11,600"},
        {"rate": "12%", "income": "$11,601 – $47,150"},
        {"rate": "22%", "income": "$47,151 – $100,525"},
        {"rate": "24%", "income": "$100,526 – $191,950"},
        {"rate": "32%", "income": "$191,951 – $243,725"},
        {"rate": "35%", "income": "$243,726 – $609,350"},
        {"rate": "37%", "income": "Over $609,350"},
    ],
    "ltcg_rates_single": [
        {"rate": "0%", "income": "Up to $47,025"},
        {"rate": "15%", "income": "$47,026 – $518,900"},
        {"rate": "20%", "income": "Over $518,900"},
    ],
    "contribution_limits_2024": {
        "401k_employee": 23000,
        "401k_catchup_50plus": 7500,
        "ira": 7000,
        "ira_catchup_50plus": 1000,
        "hsa_individual": 4150,
        "hsa_family": 8300,
        "sep_ira_max": 69000,
    },
    "roth_ira_income_limits": {
        "single_phaseout_start": 146000,
        "single_phaseout_end": 161000,
        "mfj_phaseout_start": 230000,
        "mfj_phaseout_end": 240000,
    },
    "standard_deduction": {
        "single": 14600,
        "married_filing_jointly": 29200,
        "head_of_household": 21900,
    },
}


class TaxEducationAgent(BaseAgent):
    """Agent that educates users on tax concepts, strategies, and implications for their finances."""

    name = "Tax Education Agent"
    description = (
        "I explain tax concepts relevant to investors — capital gains taxes, "
        "tax-advantaged accounts (401k, IRA, HSA, 529), tax-loss harvesting, "
        "and how to invest tax-efficiently. I use official IRS data and guidelines."
    )

    def __init__(self) -> None:
        super().__init__()
        self._retriever = get_retriever()

    def run(self, state: FinnieState) -> dict[str, Any]:
        self._logger.info("tax_education_agent_running")

        user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
        query = str(user_messages[-1].content) if user_messages else ""

        # RAG retrieval focused on tax/accounts category
        rag_context = self._retriever.get_context(query, top_k=4, category_filter="tax_accounts")
        if not rag_context:
            # Fall back to general retrieval
            rag_context = self._retriever.get_context(query, top_k=3)

        import json
        tax_json = json.dumps(TAX_DATA_2024, indent=2)

        additional_system = f"""
{self._get_user_context_str(state)}

2024 TAX REFERENCE DATA:
{tax_json}

KNOWLEDGE BASE CONTEXT:
{rag_context if rag_context else "No specific articles found for this query."}

IMPORTANT TAX DISCLAIMER:
You are providing general tax EDUCATION only. Tax situations are highly individual.
Always direct users to consult a qualified tax professional (CPA or tax attorney) for
personalized tax advice.

Provide tax education covering:
1. Answer the specific tax question clearly with 2024 data
2. Explain the underlying concept (WHY these rules exist)
3. Walk through a concrete example with numbers
4. Highlight common mistakes investors make in this area
5. Suggest related topics they should learn about

Avoid: Giving specific tax advice for their situation. Always include "consult a tax professional" reminder.
"""

        response_text = self._invoke_llm(state, additional_system)
        response_text = self._add_disclaimer(response_text)

        return {
            "messages": [AIMessage(content=response_text, name=self.name)],
            "final_response": response_text,
            "rag_context": rag_context.split("\n\n---\n\n") if rag_context else [],
        }
