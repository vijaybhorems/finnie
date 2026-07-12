"""Goals page — financial goal calculator and AI planning assistant."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agents.goal_planning_agent import _project_savings, _RETURN_ASSUMPTIONS
from src.core.config import get_settings
from src.planning.life_events import EVENT_TYPES, LifeEvent
from src.planning.projection_engine import ProjectionInputs, project_timeline, summarize
from src.workflow.graph import run_workflow


def render_goals_page() -> None:
    st.title("🎯 Financial Goal Planner")
    st.caption("Set goals, run projections, and get AI-powered planning guidance")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧮 Projection Calculator",
        "🗓️ Life Timeline",
        "🤖 AI Goal Planner",
        "💰 Retirement Calculator",
    ])

    with tab1:
        _render_projection_calculator()
    with tab2:
        _render_life_timeline()
    with tab3:
        _render_ai_goal_planner()
    with tab4:
        _render_retirement_calculator()


def _render_projection_calculator() -> None:
    st.subheader("Investment Growth Projector")

    col1, col2 = st.columns(2)

    with col1:
        current_savings = st.number_input("Current Savings ($)", min_value=0, value=10000, step=1000)
        monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=500, step=50)
        years = st.slider("Time Horizon (Years)", min_value=1, max_value=50, value=20)

    with col2:
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["conservative", "moderate", "aggressive"],
            index=1,
            key="proj_risk",
        )
        annual_return = _RETURN_ASSUMPTIONS[risk_tolerance]
        st.info(f"Assumed annual return: {annual_return*100:.1f}% ({risk_tolerance})")
        target_amount = st.number_input("Target Amount ($) — optional", min_value=0, value=0, step=10000)

    if st.button("📊 Calculate Projection", type="primary"):
        # Build projection year by year
        years_list = list(range(0, years + 1))
        conservative_values = [_project_savings(monthly_contribution, current_savings, 0.05, y) for y in years_list]
        moderate_values = [_project_savings(monthly_contribution, current_savings, 0.07, y) for y in years_list]
        aggressive_values = [_project_savings(monthly_contribution, current_savings, 0.09, y) for y in years_list]
        contributions = [current_savings + monthly_contribution * 12 * y for y in years_list]

        final_value = _project_savings(monthly_contribution, current_savings, annual_return, years)

        # Metrics
        total_contributed = current_savings + monthly_contribution * 12 * years
        growth = final_value - total_contributed
        col1, col2, col3 = st.columns(3)
        col1.metric("Projected Value", f"${final_value:,.0f}")
        col2.metric("Total Contributed", f"${total_contributed:,.0f}")
        col3.metric("Investment Growth", f"${growth:,.0f}", f"{growth/total_contributed*100:.1f}x return")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years_list, y=conservative_values, name="Conservative (5%)", line=dict(dash="dot", color="#78909C")))
        fig.add_trace(go.Scatter(x=years_list, y=moderate_values, name="Moderate (7%)", line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(x=years_list, y=aggressive_values, name="Aggressive (9%)", line=dict(dash="dot", color="#4CAF50")))
        fig.add_trace(go.Scatter(x=years_list, y=contributions, name="Total Contributed", line=dict(color="#F44336", dash="dash"), fill="tozeroy", fillcolor="rgba(244, 67, 54, 0.05)"))

        if target_amount > 0:
            fig.add_hline(y=target_amount, line_dash="dash", line_color="orange", annotation_text=f"Target: ${target_amount:,.0f}")

        fig.update_layout(
            title="Investment Growth Projection",
            xaxis_title="Years",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            height=400,
            yaxis=dict(tickprefix="$", tickformat=",.0f"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Years to target
        if target_amount > 0:
            from src.agents.goal_planning_agent import _years_to_goal
            ytg = _years_to_goal(target_amount, monthly_contribution, current_savings, annual_return)
            if ytg < float("inf"):
                st.success(f"At {annual_return*100:.1f}% annual return, you'll reach ${target_amount:,.0f} in approximately **{ytg:.1f} years**.")
            else:
                st.warning("Cannot reach target with current contribution rate. Increase monthly savings or adjust the target.")


# ── Life Timeline (deterministic life-event projection) ───────────────────────

# Per-event-kind input specs: field name -> (widget label, default, step).
_EVENT_FIELDS: dict[str, list[tuple[str, str, float, float]]] = {
    "inheritance": [("amount", "Amount ($)", 50_000, 5_000)],
    "home_purchase": [
        ("price", "Home price ($)", 400_000, 10_000),
        ("down_payment", "Down payment ($)", 80_000, 5_000),
        ("mortgage_rate", "Mortgage rate (%)", 6.0, 0.1),
        ("term_years", "Term (years)", 30, 1),
    ],
    "child_birth": [
        ("annual_cost", "Annual cost ($)", 15_000, 1_000),
        ("dependent_years", "Dependent years", 18, 1),
        ("college_start_offset", "College start (yrs from now, 0=none)", 0, 1),
        ("college_cost", "College annual cost ($)", 30_000, 5_000),
        ("college_years", "College years", 4, 1),
    ],
    "college_funding": [
        ("annual_cost", "Annual cost ($)", 30_000, 5_000),
        ("years", "Years", 4, 1),
    ],
    "job_change": [("annual_income_delta", "Annual income change ($, +/-)", 15_000, 1_000)],
    "retirement_start": [
        ("annual_retirement_spend", "Annual retirement spend ($)", 60_000, 5_000),
        ("social_security", "Annual Social Security ($)", 20_000, 1_000),
    ],
}

_EVENT_LABELS = {
    "inheritance": "Inheritance / windfall",
    "home_purchase": "Home purchase",
    "child_birth": "Child",
    "college_funding": "College funding",
    "job_change": "Job / income change",
    "retirement_start": "Retirement",
}


def _resolve_inflation(default_inflation: float) -> float:
    """Best-effort live inflation from FRED 5yr expectations; fall back to default."""
    try:
        from src.data.fred_client import FRED_SERIES, FredClient

        latest = FredClient().get_series_latest(FRED_SERIES["inflation_expectations"])
        value = latest.get("value")
        if value not in (None, ".", ""):
            return float(value) / 100.0
    except Exception:  # noqa: BLE001 — inflation is optional; keep the UI resilient
        pass
    return default_inflation


def _build_events(raw_events: list[dict]) -> list[LifeEvent]:
    """Instantiate LifeEvent objects from stored UI dicts, skipping invalid ones."""
    events: list[LifeEvent] = []
    for raw in raw_events:
        kind = raw["kind"]
        cls = EVENT_TYPES[kind]
        params = {k: v for k, v in raw.items() if k != "kind"}
        # child_birth: 0 college_start_offset means "no college block".
        if kind == "child_birth" and not params.get("college_start_offset"):
            params.pop("college_start_offset", None)
            params.pop("college_cost", None)
        events.append(cls(**params))
    return events


def _render_life_timeline() -> None:
    st.subheader("Life-Event Timeline Projection")
    st.caption("Compose major life events and see how they reshape your net worth over time")

    settings = get_settings()
    if "timeline_events" not in st.session_state:
        st.session_state.timeline_events = []

    # ── Baseline inputs ──
    col1, col2, col3 = st.columns(3)
    with col1:
        start_age = st.number_input("Current Age", min_value=18, max_value=90, value=30, key="tl_age")
        horizon = st.slider(
            "Horizon (Years)", min_value=1, max_value=settings.planning.max_horizon_years, value=30, key="tl_horizon"
        )
    with col2:
        current_savings = st.number_input("Current Savings ($)", min_value=0, value=25_000, step=5_000, key="tl_savings")
        monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=1_000, step=100, key="tl_contrib")
    with col3:
        risk_tolerance = st.selectbox(
            "Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1, key="tl_risk"
        )
        annual_return = _RETURN_ASSUMPTIONS[risk_tolerance]
        view_real = st.toggle("Inflation-adjusted (real $)", value=False, key="tl_real")
        st.caption(f"Assumed return: {annual_return*100:.1f}%")

    # ── Event builder ──
    st.markdown("**Add a life event**")
    ecol1, ecol2 = st.columns([1, 3])
    with ecol1:
        kind = st.selectbox(
            "Event type",
            list(_EVENT_FIELDS.keys()),
            format_func=lambda k: _EVENT_LABELS.get(k, k),
            key="tl_kind",
        )
    with ecol2:
        year_offset = st.slider("Years from now", min_value=0, max_value=int(horizon), value=1, key="tl_offset")

    new_params: dict = {}
    field_cols = st.columns(len(_EVENT_FIELDS[kind]))
    for (field_name, label, default, step), fcol in zip(_EVENT_FIELDS[kind], field_cols):
        new_params[field_name] = fcol.number_input(
            label, value=default, step=step, key=f"tl_field_{kind}_{field_name}"
        )

    if st.button("➕ Add event", key="tl_add"):
        if len(st.session_state.timeline_events) >= settings.planning.max_events:
            st.warning(f"Event limit reached ({settings.planning.max_events}).")
        else:
            entry = {"kind": kind, "year_offset": int(year_offset), "label": _EVENT_LABELS.get(kind, kind)}
            for field_name, _, _, _ in _EVENT_FIELDS[kind]:
                val = new_params[field_name]
                # Rate field is entered as a percent; convert to a fraction.
                entry[field_name] = val / 100.0 if field_name == "mortgage_rate" else val
            st.session_state.timeline_events.append(entry)
            st.rerun()

    # ── Current timeline (sorted by year) ──
    events_raw = sorted(st.session_state.timeline_events, key=lambda e: e["year_offset"])
    if events_raw:
        st.markdown("**Your timeline**")
        for i, evt in enumerate(events_raw):
            c1, c2 = st.columns([6, 1])
            c1.write(f"• Year +{evt['year_offset']} (age {start_age + evt['year_offset']}): {evt['label']}")
            if c2.button("🗑️", key=f"tl_del_{i}"):
                st.session_state.timeline_events.remove(evt)
                st.rerun()
        if st.button("Clear all events", key="tl_clear"):
            st.session_state.timeline_events = []
            st.rerun()

    if not st.button("📈 Run Timeline Projection", type="primary", key="tl_run"):
        return

    # ── Run projection ──
    inflation = _resolve_inflation(settings.planning.default_inflation)
    inputs = ProjectionInputs(
        start_age=int(start_age),
        horizon_years=int(horizon),
        current_savings=float(current_savings),
        monthly_contribution=float(monthly_contribution),
        annual_return=annual_return,
        annual_inflation=inflation,
    )
    try:
        events = _build_events(events_raw)
    except Exception as exc:  # noqa: BLE001 — surface bad inputs to the user
        st.error(f"Could not build the timeline: {exc}")
        return

    baseline = project_timeline(inputs, [])
    scenario = project_timeline(inputs, events)
    summary = summarize(scenario, annual_inflation=inflation)

    def _series(results: list) -> list[float]:
        if not view_real:
            return [r.net_worth for r in results]
        return [r.net_worth / ((1 + inflation) ** r.year) for r in results]

    ages = [r.age for r in scenario]

    unit = "real $" if view_real else "nominal $"
    m1, m2, m3 = st.columns(3)
    ending = summary["ending_net_worth_real"] if view_real else summary["ending_net_worth"]
    m1.metric(f"Ending Net Worth ({unit})", f"${ending:,.0f}")
    m2.metric("Total Contributed", f"${summary['total_contributed']:,.0f}")
    m3.metric("Investment Growth", f"${summary['investment_growth']:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=_series(baseline), name="Baseline (no events)", line=dict(color="#78909C", dash="dot")))
    fig.add_trace(go.Scatter(x=ages, y=_series(scenario), name="With life events", line=dict(color="#2196F3"), fill="tozeroy", fillcolor="rgba(33, 150, 243, 0.08)"))
    for evt in events_raw:
        fig.add_vline(x=start_age + evt["year_offset"], line_dash="dash", line_color="rgba(244, 67, 54, 0.4)", annotation_text=evt["label"], annotation_textangle=-90)
    fig.update_layout(
        title=f"Net Worth Over Time ({unit})",
        xaxis_title="Age",
        yaxis_title=f"Net Worth ({unit})",
        hovermode="x unified",
        height=420,
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if summary["milestones"]:
        st.markdown("**Milestones**")
        milestone_df = pd.DataFrame([
            {
                "Year": m["year"],
                "Age": m["age"],
                "Event(s)": ", ".join(m["events"]),
                "Net Worth": f"${m['net_worth']:,.0f}",
            }
            for m in summary["milestones"]
        ])
        st.dataframe(milestone_df, use_container_width=True, hide_index=True)

    # ── Optional AI narration (reuses GoalPlanningAgent via the workflow) ──
    if st.button("🤖 Explain this projection", key="tl_explain"):
        event_lines = "; ".join(
            f"{e['label']} at age {start_age + e['year_offset']}" for e in events_raw
        ) or "no specific events"
        summary_msg = (
            f"I modeled a financial timeline from age {start_age} over {horizon} years "
            f"with these life events: {event_lines}. Starting savings ${current_savings:,.0f}, "
            f"monthly contribution ${monthly_contribution:,.0f}, {risk_tolerance} risk "
            f"({annual_return*100:.1f}% return). Projected ending net worth is "
            f"${summary['ending_net_worth']:,.0f} nominal (${summary['ending_net_worth_real']:,.0f} "
            f"in today's dollars). Explain what drives this trajectory and how the life events "
            f"affect it, educationally."
        )
        with st.spinner("Finnie is analyzing your timeline..."):
            result = run_workflow(
                user_message=summary_msg,
                user_profile=st.session_state.get("user_profile"),
            )
        st.markdown(result["final_response"])


def _render_ai_goal_planner() -> None:
    st.subheader("AI Goal Planning Assistant")
    st.caption("Describe your financial goals and get personalized planning guidance")

    goal_templates = [
        "I want to retire at age 65 with $80,000/year income. I'm 35 years old with $50,000 saved.",
        "I want to save for a $60,000 down payment on a home in 5 years. I have $10,000 saved.",
        "I want to build 6 months emergency fund. My monthly expenses are $4,000.",
        "I want to achieve financial independence and retire early at 50. I'm 30 with $100,000 saved.",
        "I want to save $200,000 for my child's college education. They are 5 years old.",
    ]

    selected_template = st.selectbox("Goal Templates", [""] + goal_templates, key="goal_template")
    custom_goal = st.text_area("Describe your goal", value=selected_template, height=100, key="custom_goal")

    risk_tolerance = st.session_state.get("user_profile", {}).get("risk_tolerance", "moderate")

    if custom_goal and st.button("🎯 Create My Goal Plan", type="primary"):
        with st.spinner("Building your personalized goal plan..."):
            result = run_workflow(
                user_message=custom_goal,
                user_profile=st.session_state.get("user_profile"),
            )
        st.markdown(result["final_response"])


def _render_retirement_calculator() -> None:
    st.subheader("Retirement Calculator")
    st.caption("Estimate how much you need and whether you're on track")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Your Information**")
        current_age = st.number_input("Current Age", min_value=18, max_value=80, value=30)
        retirement_age = st.number_input("Target Retirement Age", min_value=current_age + 1, max_value=90, value=65)
        current_savings = st.number_input("Current Retirement Savings ($)", min_value=0, value=50000, step=5000)
        monthly_savings = st.number_input("Monthly Retirement Savings ($)", min_value=0, value=1000, step=100)

    with col2:
        st.markdown("**Retirement Assumptions**")
        annual_expenses = st.number_input("Desired Annual Retirement Income ($)", min_value=0, value=60000, step=5000)
        social_security = st.number_input("Expected Annual Social Security ($)", min_value=0, value=20000, step=1000)
        risk_tolerance = st.selectbox("Risk Profile", ["conservative", "moderate", "aggressive"], index=1, key="ret_risk")

    if st.button("🧮 Calculate Retirement Readiness", type="primary"):
        years_to_retire = retirement_age - current_age
        annual_return = _RETURN_ASSUMPTIONS[risk_tolerance]

        # Projected portfolio at retirement
        projected = _project_savings(monthly_savings, current_savings, annual_return, years_to_retire)

        # Required portfolio (4% rule)
        net_annual_need = max(0, annual_expenses - social_security)
        required_portfolio = net_annual_need * 25  # 4% rule

        # 30-year retirement spending (inflation-adjusted at 3%)
        col1, col2, col3 = st.columns(3)
        col1.metric("Projected Portfolio", f"${projected:,.0f}")
        col2.metric("Required Portfolio (4% rule)", f"${required_portfolio:,.0f}")

        gap = projected - required_portfolio
        if gap >= 0:
            col3.metric("Status", "✅ On Track", f"+${gap:,.0f} surplus")
        else:
            col3.metric("Status", "⚠️ Gap Exists", f"-${abs(gap):,.0f} shortfall")

        # Required monthly savings to hit goal
        if required_portfolio > projected:
            needed_additional = _calculate_required_monthly(required_portfolio, current_savings, annual_return, years_to_retire) - monthly_savings
            if needed_additional > 0:
                st.warning(f"To close the gap, consider increasing monthly savings by ~${needed_additional:,.0f}/month.")

        # Projection chart
        years_list = list(range(0, years_to_retire + 1))
        values = [_project_savings(monthly_savings, current_savings, annual_return, y) for y in years_list]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[current_age + y for y in years_list], y=values, name="Projected Portfolio", fill="tozeroy", fillcolor="rgba(33, 150, 243, 0.1)", line=dict(color="#2196F3")))
        fig.add_hline(y=required_portfolio, line_dash="dash", line_color="orange", annotation_text=f"Required: ${required_portfolio:,.0f}")
        fig.update_layout(title="Retirement Portfolio Projection", xaxis_title="Age", yaxis_title="Portfolio Value ($)", height=350, yaxis=dict(tickprefix="$", tickformat=",.0f"))
        st.plotly_chart(fig, use_container_width=True)


def _calculate_required_monthly(target: float, current: float, annual_return: float, years: int) -> float:
    """Calculate monthly contribution needed to reach target."""
    r = annual_return / 12
    n = years * 12
    fv_current = current * (1 + r) ** n
    remaining = target - fv_current
    if remaining <= 0:
        return 0.0
    if r <= 0:
        return remaining / n
    return remaining * r / ((1 + r) ** n - 1)
