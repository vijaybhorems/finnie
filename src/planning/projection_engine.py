"""Deterministic year-by-year life-event projection engine.

Generalizes the single-goal annuity math in
``src.agents.goal_planning_agent._project_savings`` to a multi-event timeline.
With an empty event list the per-year net worth reproduces ``_project_savings``
exactly (monthly compounding of a lump sum + level contributions).

Pure functions only — no I/O. Inflation is an explicit input so the engine stays
deterministic and testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.planning.life_events import LifeEvent


@dataclass
class ProjectionInputs:
    """Baseline parameters for a projection."""

    start_age: int
    horizon_years: int
    current_savings: float
    monthly_contribution: float
    annual_return: float
    annual_inflation: float = 0.03


@dataclass
class YearResult:
    """State at the end of a projected year (year 0 = starting point)."""

    year: int
    age: int
    net_worth: float
    cash_flow: float          # net savings added this year (nominal)
    contributions: float      # cumulative principal in (excludes growth)
    events: list[str] = field(default_factory=list)


def _aggregate_deltas(
    events: list[LifeEvent], horizon_years: int, base_monthly_contribution: float
) -> tuple[dict[int, float], dict[int, float]]:
    """Sum each event's savings and one-time deltas into per-year maps."""
    savings: dict[int, float] = {}
    one_time: dict[int, float] = {}
    for event in events:
        ev_savings, ev_one_time = event.deltas(horizon_years, base_monthly_contribution)
        for idx, amount in ev_savings.items():
            if 0 <= idx < horizon_years:
                savings[idx] = savings.get(idx, 0.0) + amount
        for idx, amount in ev_one_time.items():
            if 0 <= idx < horizon_years:
                one_time[idx] = one_time.get(idx, 0.0) + amount
    return savings, one_time


def project_timeline(
    inputs: ProjectionInputs,
    events: list[LifeEvent] | None = None,
) -> list[YearResult]:
    """Project net worth / cash flow year by year with events applied in order."""
    events = events or []
    horizon = inputs.horizon_years
    savings_by_year, one_time_by_year = _aggregate_deltas(
        events, horizon, inputs.monthly_contribution
    )

    r = inputs.annual_return / 12
    base_annual = inputs.monthly_contribution * 12

    balance = inputs.current_savings
    cumulative = inputs.current_savings
    results = [
        YearResult(
            year=0,
            age=inputs.start_age,
            net_worth=round(balance, 2),
            cash_flow=0.0,
            contributions=round(cumulative, 2),
            events=[],
        )
    ]

    for year in range(1, horizon + 1):
        idx = year - 1
        net_annual = base_annual + savings_by_year.get(idx, 0.0)
        one_time = one_time_by_year.get(idx, 0.0)
        effective_monthly = net_annual / 12

        for _ in range(12):
            balance = balance * (1 + r) + effective_monthly
        balance += one_time

        cumulative += net_annual + one_time
        labels = [e.display() for e in events if e.year_offset == idx]
        results.append(
            YearResult(
                year=year,
                age=inputs.start_age + year,
                net_worth=round(balance, 2),
                cash_flow=round(net_annual, 2),
                contributions=round(cumulative, 2),
                events=labels,
            )
        )

    return results


def summarize(results: list[YearResult], annual_inflation: float = 0.03) -> dict:
    """Milestone table + ending totals (nominal & inflation-adjusted real)."""
    if not results:
        return {}

    ending = results[-1]
    horizon = ending.year
    deflator = (1 + annual_inflation) ** horizon if horizon else 1.0
    real_net_worth = ending.net_worth / deflator if deflator else ending.net_worth

    milestones = [
        {
            "year": r.year,
            "age": r.age,
            "events": r.events,
            "net_worth": r.net_worth,
        }
        for r in results
        if r.events
    ]

    return {
        "ending_net_worth": round(ending.net_worth, 2),
        "ending_net_worth_real": round(real_net_worth, 2),
        "total_contributed": round(ending.contributions, 2),
        "investment_growth": round(ending.net_worth - ending.contributions, 2),
        "horizon_years": horizon,
        "milestones": milestones,
    }
