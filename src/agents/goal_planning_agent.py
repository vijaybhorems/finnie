"""Goal Planning Agent — financial goal setting with risk-adjusted projections."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.core.state import FinnieState
from src.data.fred_client import FredClient


def _project_savings(
    monthly_contribution: float,
    current_savings: float,
    annual_return: float,
    years: int,
) -> float:
    """Future value of regular contributions + lump sum."""
    r = annual_return / 12
    n = years * 12
    fv_contributions = monthly_contribution * ((1 + r) ** n - 1) / r if r > 0 else monthly_contribution * n
    fv_lump = current_savings * (1 + r) ** n
    return round(fv_lump + fv_contributions, 2)


def _years_to_goal(
    target: float,
    monthly_contribution: float,
    current_savings: float,
    annual_return: float,
) -> float:
    """Estimate years needed to reach a savings target."""
    import math
    r = annual_return / 12
    if r <= 0:
        if monthly_contribution <= 0:
            return float("inf")
        return (target - current_savings) / monthly_contribution / 12

    # Newton's method approximation
    for years in range(1, 100):
        projected = _project_savings(monthly_contribution, current_savings, annual_return, years)
        if projected >= target:
            return years
    return float("inf")


_RETURN_ASSUMPTIONS = {
    "conservative": 0.05,   # 60/40 portfolio
    "moderate": 0.07,       # classic balanced portfolio
    "aggressive": 0.09,     # stock-heavy portfolio
}


class GoalPlanningAgent(BaseAgent):
    """Agent that helps users set and plan financial goals with projections based on their profile."""

    name = "Goal Planning Agent"
    description = (
        "I help you set and plan financial goals — retirement, home purchase, education, "
        "or financial independence. I provide projections based on your risk tolerance, "
        "time horizon, and current economic conditions."
    )

    def __init__(self) -> None:
        super().__init__()
        self._fred = FredClient()

    def run(self, state: FinnieState) -> dict[str, Any]:
        self._logger.info("goal_planning_agent_running")

        rate_env = self._fred.get_interest_rate_environment()
        rate_env_str = json.dumps(rate_env, indent=2, default=str)

        profile = state.user_profile
        annual_return = _RETURN_ASSUMPTIONS.get(profile.risk_tolerance, 0.07)

        # Build projection examples
        projections = self._build_projection_examples(annual_return, profile.risk_tolerance)

        additional_system = f"""
{self._get_user_context_str(state)}

CURRENT ECONOMIC ENVIRONMENT (from FRED):
{rate_env_str}

RETURN ASSUMPTIONS FOR {profile.risk_tolerance.upper()} INVESTOR:
- Expected annual return: {annual_return*100:.1f}%
- Note: These are historical averages; actual returns will vary

PROJECTION EXAMPLES FOR PLANNING PURPOSES:
{projections}

Help the user with financial goal planning:
1. If they mention a specific goal (retirement, home, education), calculate time and required savings
2. Explain how risk tolerance affects achievable outcomes
3. Discuss how the current interest rate environment affects savings and investment returns
4. Break down the goal into actionable monthly steps
5. Mention tax-advantaged accounts appropriate for the goal

Ask clarifying questions if needed: target amount, current savings, monthly contribution ability, timeline.
"""

        response_text = self._invoke_llm(state, additional_system)
        response_text = self._add_disclaimer(response_text)

        return {
            "messages": [AIMessage(content=response_text, name=self.name)],
            "final_response": response_text,
        }

    def _build_projection_examples(self, annual_return: float, risk_level: str) -> str:
        scenarios = [
            (500, 0, 10),
            (500, 0, 20),
            (500, 0, 30),
            (1000, 10000, 20),
            (1500, 50000, 10),
        ]
        lines = [f"Monthly Contrib | Start Savings | Years | Projected Value ({risk_level}, {annual_return*100:.1f}%/yr)"]
        lines.append("-" * 80)
        for contrib, start, years in scenarios:
            fv = _project_savings(contrib, start, annual_return, years)
            lines.append(f"${contrib:,}/mo    | ${start:,}         | {years}y   | ${fv:,.0f}")
        return "\n".join(lines)
