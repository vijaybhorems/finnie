"""Unit tests for the life-event schema and delta contracts."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.planning.life_events import (
    EVENT_TYPES,
    ChildBirth,
    CollegeFunding,
    HomePurchase,
    Inheritance,
    JobChange,
    RetirementStart,
    _annual_mortgage_payment,
)


class TestCatalog:
    def test_event_types_registry_complete(self):
        assert set(EVENT_TYPES) == {
            "inheritance",
            "home_purchase",
            "child_birth",
            "college_funding",
            "job_change",
            "retirement_start",
        }
        for kind, cls in EVENT_TYPES.items():
            assert cls.model_fields["kind"].default == kind


class TestValidation:
    def test_negative_year_offset_rejected(self):
        with pytest.raises(ValidationError):
            Inheritance(year_offset=-1, amount=1000)

    def test_non_positive_inheritance_rejected(self):
        with pytest.raises(ValidationError):
            Inheritance(year_offset=0, amount=0)

    def test_home_purchase_requires_positive_term(self):
        with pytest.raises(ValidationError):
            HomePurchase(year_offset=0, price=100, down_payment=0, mortgage_rate=0.05, term_years=0)


class TestDeltas:
    def test_inheritance_one_time_only(self):
        savings, one_time = Inheritance(year_offset=2, amount=5000).deltas(10, 500)
        assert savings == {}
        assert one_time == {2: 5000}

    def test_deltas_clamped_by_horizon_in_event(self):
        # College span extends past horizon; event returns only in-range indices.
        savings, _ = CollegeFunding(year_offset=8, annual_cost=1000, years=5).deltas(10, 500)
        assert set(savings) == {8, 9}

    def test_job_change_runs_to_horizon(self):
        savings, _ = JobChange(year_offset=3, annual_income_delta=1000).deltas(6, 500)
        assert savings == {3: 1000, 4: 1000, 5: 1000}

    def test_retirement_cancels_base_and_adds_net_spend(self):
        savings, _ = RetirementStart(
            year_offset=1, annual_retirement_spend=40000, social_security=10000
        ).deltas(3, 1000)
        # base 12k cancel + 30k net spend = -42k each year from offset.
        assert savings == {1: -42000, 2: -42000}

    def test_child_dependent_and_college_windows(self):
        savings, _ = ChildBirth(
            year_offset=0,
            annual_cost=10000,
            dependent_years=2,
            college_start_offset=5,
            college_cost=20000,
            college_years=2,
        ).deltas(10, 500)
        assert savings[0] == -10000
        assert savings[1] == -10000
        assert 2 not in savings  # dependent window ended
        assert savings[5] == -20000
        assert savings[6] == -20000


class TestMortgageMath:
    def test_zero_rate_amortization(self):
        assert _annual_mortgage_payment(120_000, 0.0, 10) == pytest.approx(12_000)

    def test_positive_rate_amortization_reasonable(self):
        # 30-yr, 240k @ 6% ~ $1,438.9/mo -> ~$17,267/yr.
        annual = _annual_mortgage_payment(240_000, 0.06, 30)
        assert annual == pytest.approx(17_267, abs=100)
