"""Life-event schema and closed v1 catalog.

Each event maps itself onto two per-year delta channels the projection engine
understands:

  - ``savings_delta``  : annual change to net savings (income positive,
                         expense/outflow negative). Folded into the effective
                         monthly contribution for that year.
  - ``one_time_delta`` : a lump-sum net-worth change applied once, at the end of
                         the event's year.

Deltas are returned as ``{year_index: amount}`` maps keyed on the year offset
from the projection start (year 0 = the first projected year). The engine clamps
indices to the projection horizon.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Number of years a child is treated as a dependent by default.
_DEFAULT_DEPENDENT_YEARS = 18


def _annual_mortgage_payment(principal: float, annual_rate: float, term_years: int) -> float:
    """Standard fixed-rate amortized annual payment (12 monthly payments)."""
    if term_years <= 0:
        return 0.0
    if annual_rate <= 0:
        return principal / term_years
    r = annual_rate / 12
    n = term_years * 12
    monthly = principal * r / (1 - (1 + r) ** -n)
    return monthly * 12


class LifeEvent(BaseModel):
    """Base life event applying deltas at a given year offset from start."""

    kind: str
    label: str = ""
    year_offset: int = Field(ge=0)

    def deltas(
        self, horizon_years: int, base_monthly_contribution: float
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Return ``(savings_delta_by_year, one_time_by_year)`` for this event.

        ``base_monthly_contribution`` is provided so events like retirement can
        cancel ongoing contributions.
        """
        raise NotImplementedError

    def display(self) -> str:
        """Human-readable label for charts/milestones."""
        return self.label or self.kind.replace("_", " ").title()


class Inheritance(LifeEvent):
    """One-time asset inflow."""

    kind: Literal["inheritance"] = "inheritance"
    amount: float = Field(gt=0)

    def deltas(self, horizon_years, base_monthly_contribution):
        return {}, {self.year_offset: self.amount}


class HomePurchase(LifeEvent):
    """Down payment (one-time) plus an amortized mortgage outflow (recurring)."""

    kind: Literal["home_purchase"] = "home_purchase"
    price: float = Field(gt=0)
    down_payment: float = Field(ge=0)
    mortgage_rate: float = Field(ge=0)
    term_years: int = Field(gt=0)

    def deltas(self, horizon_years, base_monthly_contribution):
        one_time = {self.year_offset: -self.down_payment}
        principal = max(0.0, self.price - self.down_payment)
        annual_payment = _annual_mortgage_payment(principal, self.mortgage_rate, self.term_years)
        savings: dict[int, float] = {}
        for y in range(self.year_offset, min(self.year_offset + self.term_years, horizon_years)):
            savings[y] = savings.get(y, 0.0) - annual_payment
        return savings, one_time


class ChildBirth(LifeEvent):
    """Recurring child-rearing cost, with an optional later college block."""

    kind: Literal["child_birth"] = "child_birth"
    annual_cost: float = Field(ge=0)
    dependent_years: int = Field(default=_DEFAULT_DEPENDENT_YEARS, gt=0)
    college_start_offset: int | None = None
    college_cost: float | None = None
    college_years: int = Field(default=4, gt=0)

    def deltas(self, horizon_years, base_monthly_contribution):
        savings: dict[int, float] = {}
        for y in range(self.year_offset, min(self.year_offset + self.dependent_years, horizon_years)):
            savings[y] = savings.get(y, 0.0) - self.annual_cost
        if self.college_start_offset is not None and self.college_cost:
            for y in range(
                self.college_start_offset,
                min(self.college_start_offset + self.college_years, horizon_years),
            ):
                savings[y] = savings.get(y, 0.0) - self.college_cost
        return savings, {}


class CollegeFunding(LifeEvent):
    """Standalone recurring education outflow over a fixed number of years."""

    kind: Literal["college_funding"] = "college_funding"
    annual_cost: float = Field(gt=0)
    years: int = Field(gt=0)

    def deltas(self, horizon_years, base_monthly_contribution):
        savings: dict[int, float] = {}
        for y in range(self.year_offset, min(self.year_offset + self.years, horizon_years)):
            savings[y] = savings.get(y, 0.0) - self.annual_cost
        return savings, {}


class JobChange(LifeEvent):
    """Step change in annual income, assumed to flow to net savings."""

    kind: Literal["job_change"] = "job_change"
    annual_income_delta: float

    def deltas(self, horizon_years, base_monthly_contribution):
        savings = {
            y: self.annual_income_delta for y in range(self.year_offset, horizon_years)
        }
        return savings, {}


class RetirementStart(LifeEvent):
    """Stops ongoing contributions and begins net drawdown for spending."""

    kind: Literal["retirement_start"] = "retirement_start"
    annual_retirement_spend: float = Field(ge=0)
    social_security: float = Field(default=0.0, ge=0)

    def deltas(self, horizon_years, base_monthly_contribution):
        base_annual = base_monthly_contribution * 12
        net_spend = self.annual_retirement_spend - self.social_security
        savings = {
            y: -base_annual - net_spend for y in range(self.year_offset, horizon_years)
        }
        return savings, {}


# Closed v1 catalog, keyed by ``kind`` for UI construction / validation.
EVENT_TYPES: dict[str, type[LifeEvent]] = {
    cls.model_fields["kind"].default: cls
    for cls in (
        Inheritance,
        HomePurchase,
        ChildBirth,
        CollegeFunding,
        JobChange,
        RetirementStart,
    )
}
