"""Unit tests for the deterministic life-event projection engine."""
from __future__ import annotations

import pytest

from src.agents.goal_planning_agent import _project_savings
from src.planning.life_events import (
    ChildBirth,
    HomePurchase,
    Inheritance,
    JobChange,
    RetirementStart,
)
from src.planning.projection_engine import ProjectionInputs, project_timeline, summarize


def _inputs(**kw) -> ProjectionInputs:
    defaults = dict(
        start_age=30,
        horizon_years=20,
        current_savings=10_000.0,
        monthly_contribution=500.0,
        annual_return=0.07,
        annual_inflation=0.03,
    )
    defaults.update(kw)
    return ProjectionInputs(**defaults)


class TestBaselineParity:
    """Empty timeline reproduces the existing _project_savings math per year."""

    def test_empty_timeline_matches_project_savings(self):
        inp = _inputs()
        results = project_timeline(inp, [])
        for r in results:
            expected = _project_savings(
                inp.monthly_contribution, inp.current_savings, inp.annual_return, r.year
            )
            assert r.net_worth == pytest.approx(expected, abs=0.05), (
                f"year {r.year}: {r.net_worth} != {expected}"
            )

    def test_year_zero_is_starting_point(self):
        inp = _inputs()
        results = project_timeline(inp, [])
        assert results[0].year == 0
        assert results[0].age == 30
        assert results[0].net_worth == 10_000.0

    def test_result_length_covers_horizon(self):
        results = project_timeline(_inputs(horizon_years=15), [])
        assert len(results) == 16  # year 0 + 15 projected years


class TestOneTimeEvents:
    def test_inheritance_adds_lump_at_year(self):
        inp = _inputs(horizon_years=10)
        base = project_timeline(inp, [])
        with_evt = project_timeline(inp, [Inheritance(year_offset=4, amount=50_000)])
        # Years before the event are unchanged.
        assert with_evt[4].net_worth == pytest.approx(base[4].net_worth, abs=0.05)
        # The event year jumps by the lump (applied at year end, no growth yet).
        assert with_evt[5].net_worth == pytest.approx(base[5].net_worth + 50_000, abs=0.05)
        # Later the lump compounds, so the gap grows beyond the raw amount.
        assert with_evt[9].net_worth > base[9].net_worth + 50_000


class TestRecurringEvents:
    def test_job_change_steps_savings(self):
        inp = _inputs(horizon_years=10)
        base = project_timeline(inp, [])
        raise_evt = project_timeline(inp, [JobChange(year_offset=3, annual_income_delta=12_000)])
        # Cash flow steps up from the event year onward.
        assert raise_evt[3].cash_flow == pytest.approx(base[3].cash_flow, abs=0.01)
        assert raise_evt[4].cash_flow == pytest.approx(base[4].cash_flow + 12_000, abs=0.01)
        assert raise_evt[10].net_worth > base[10].net_worth

    def test_home_purchase_down_payment_and_mortgage(self):
        inp = _inputs(horizon_years=10)
        base = project_timeline(inp, [])
        evt = HomePurchase(
            year_offset=2, price=300_000, down_payment=60_000, mortgage_rate=0.06, term_years=30
        )
        with_evt = project_timeline(inp, [evt])
        # Net worth drops relative to baseline after the purchase (down payment + mortgage).
        assert with_evt[3].net_worth < base[3].net_worth
        # Mortgage reduces annual net savings during the term.
        assert with_evt[5].cash_flow < base[5].cash_flow

    def test_zero_rate_mortgage_is_principal_over_term(self):
        inp = _inputs(horizon_years=6, monthly_contribution=0, annual_return=0, current_savings=0)
        evt = HomePurchase(
            year_offset=0, price=120_000, down_payment=0, mortgage_rate=0.0, term_years=5
        )
        results = project_timeline(inp, [evt])
        # 120k over 5 years at 0% = 24k/yr outflow; net worth after 5 yrs = -120k.
        assert results[5].net_worth == pytest.approx(-120_000, abs=0.05)

    def test_retirement_stops_contributions_and_draws_down(self):
        inp = _inputs(horizon_years=10, monthly_contribution=1_000, annual_return=0.0)
        evt = RetirementStart(year_offset=5, annual_retirement_spend=40_000, social_security=10_000)
        results = project_timeline(inp, [evt])
        # Base contributions (12k) stop and net spend (40k-10k=30k) is withdrawn,
        # so net annual savings = 12k base + (-12k - 30k) delta = -30k drawdown.
        assert results[6].cash_flow == pytest.approx(-30_000, abs=0.01)
        assert results[10].net_worth < results[5].net_worth

    def test_child_birth_dependent_and_college(self):
        inp = _inputs(horizon_years=25)
        base = project_timeline(inp, [])
        evt = ChildBirth(
            year_offset=1,
            annual_cost=12_000,
            dependent_years=18,
            college_start_offset=19,
            college_cost=30_000,
            college_years=4,
        )
        with_evt = project_timeline(inp, [evt])
        # Dependent cost lowers savings during childhood.
        assert with_evt[5].cash_flow == pytest.approx(base[5].cash_flow - 12_000, abs=0.01)
        # College block lowers savings further in its window (year index 19 -> year 20).
        assert with_evt[20].cash_flow == pytest.approx(base[20].cash_flow - 30_000, abs=0.01)


class TestOrdering:
    def test_out_of_order_events_applied_by_offset(self):
        inp = _inputs(horizon_years=8)
        ordered = project_timeline(
            inp, [Inheritance(year_offset=1, amount=10_000), Inheritance(year_offset=5, amount=20_000)]
        )
        reversed_order = project_timeline(
            inp, [Inheritance(year_offset=5, amount=20_000), Inheritance(year_offset=1, amount=10_000)]
        )
        assert [r.net_worth for r in ordered] == [r.net_worth for r in reversed_order]

    def test_two_events_same_year_both_applied(self):
        inp = _inputs(horizon_years=6)
        base = project_timeline(inp, [])
        evt = project_timeline(
            inp, [Inheritance(year_offset=2, amount=10_000), Inheritance(year_offset=2, amount=5_000)]
        )
        assert evt[3].net_worth == pytest.approx(base[3].net_worth + 15_000, abs=0.05)


class TestSummarize:
    def test_summary_real_and_milestones(self):
        inp = _inputs(horizon_years=10)
        results = project_timeline(inp, [Inheritance(year_offset=3, amount=25_000, label="Windfall")])
        summary = summarize(results, annual_inflation=inp.annual_inflation)
        assert summary["horizon_years"] == 10
        assert summary["ending_net_worth_real"] < summary["ending_net_worth"]
        assert summary["ending_net_worth"] == results[-1].net_worth
        # One milestone row for the single event year.
        assert len(summary["milestones"]) == 1
        assert summary["milestones"][0]["year"] == 4
        assert "Windfall" in summary["milestones"][0]["events"]

    def test_empty_results_summary(self):
        assert summarize([]) == {}
