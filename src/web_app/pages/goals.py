"""Goals page — financial goal calculator and AI planning assistant."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agents.goal_planning_agent import _project_savings, _RETURN_ASSUMPTIONS
from src.workflow.graph import run_workflow


def render_goals_page() -> None:
    st.title("🎯 Financial Goal Planner")
    st.caption("Set goals, run projections, and get AI-powered planning guidance")

    tab1, tab2, tab3 = st.tabs(["🧮 Projection Calculator", "🤖 AI Goal Planner", "💰 Retirement Calculator"])

    with tab1:
        _render_projection_calculator()
    with tab2:
        _render_ai_goal_planner()
    with tab3:
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
