"""Portfolio analysis page — interactive portfolio input with visualizations."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.yfinance_client import YFinanceClient
from src.workflow.graph import run_workflow


def render_portfolio_page() -> None:
    st.title("📊 Portfolio Analysis")
    st.caption("Enter your holdings for an AI-powered portfolio analysis")

    tab1, tab2 = st.tabs(["📋 Holdings Input", "🤖 AI Analysis"])

    with tab1:
        _render_holdings_input()

    with tab2:
        _render_ai_analysis()


def _render_holdings_input() -> None:
    st.subheader("Your Holdings")

    if "portfolio_holdings" not in st.session_state:
        st.session_state.portfolio_holdings = [
            {"ticker": "AAPL", "shares": 10.0, "avg_cost": 150.0},
            {"ticker": "MSFT", "shares": 5.0, "avg_cost": 300.0},
            {"ticker": "VTI", "shares": 20.0, "avg_cost": 220.0},
        ]

    # Editable table
    holdings_df = pd.DataFrame(st.session_state.portfolio_holdings)
    edited_df = st.data_editor(
        holdings_df,
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", max_chars=5),
            "shares": st.column_config.NumberColumn("Shares", min_value=0.0, format="%.2f"),
            "avg_cost": st.column_config.NumberColumn("Avg Cost ($)", min_value=0.0, format="$%.2f"),
        },
        use_container_width=True,
        key="holdings_editor",
    )

    col1, col2 = st.columns(2)
    if col1.button("📊 Fetch Current Data", type="primary", use_container_width=True):
        holdings = edited_df.to_dict(orient="records")
        st.session_state.portfolio_holdings = holdings
        with st.spinner("Fetching market data..."):
            _display_portfolio_metrics(holdings)

    if col2.button("🤖 Get AI Analysis", type="secondary", use_container_width=True):
        holdings = edited_df.to_dict(orient="records")
        st.session_state.portfolio_holdings = holdings
        st.session_state.portfolio_ai_requested = True
        st.rerun()

    # Show portfolio metrics if already fetched
    if "portfolio_metrics" in st.session_state:
        _render_portfolio_charts(st.session_state.portfolio_metrics)


def _display_portfolio_metrics(holdings: list[dict]) -> None:
    yf_client = YFinanceClient()
    metrics = yf_client.get_portfolio_metrics(holdings)
    st.session_state.portfolio_metrics = metrics

    if "error" in metrics:
        st.error(f"Error fetching data: {metrics['error']}")
        return

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", f"${metrics['total_value']:,.2f}")
    col2.metric("Total Cost", f"${metrics['total_cost']:,.2f}")

    gain = metrics["total_gain_loss"]
    gain_pct = metrics["total_gain_loss_pct"]
    col3.metric("Total Gain/Loss", f"${gain:,.2f}", f"{gain_pct:+.2f}%")
    col4.metric("Holdings", len(metrics.get("holdings", [])))

    _render_portfolio_charts(metrics)


def _render_portfolio_charts(metrics: dict) -> None:
    holdings_list = metrics.get("holdings", [])
    if not holdings_list:
        return

    df = pd.DataFrame(holdings_list)

    # Allocation pie chart
    fig_alloc = px.pie(
        df,
        values="position_value",
        names="ticker",
        title="Portfolio Allocation by Value",
        hole=0.3,
    )
    fig_alloc.update_traces(textposition="inside", textinfo="percent+label")

    # Gain/Loss bar chart
    colors = ["#26a69a" if x >= 0 else "#ef5350" for x in df["gain_loss"]]
    fig_gl = go.Figure(go.Bar(
        x=df["ticker"],
        y=df["gain_loss_pct"],
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["gain_loss_pct"]],
        textposition="outside",
    ))
    fig_gl.update_layout(
        title="Gain/Loss % by Position",
        yaxis_title="Gain/Loss %",
        showlegend=False,
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_alloc, use_container_width=True)
    col2.plotly_chart(fig_gl, use_container_width=True)

    # Sector breakdown
    sector_df = df[df["sector"].notna()].copy()
    if not sector_df.empty:
        sector_values = sector_df.groupby("sector")["position_value"].sum().reset_index()
        fig_sector = px.bar(
            sector_values,
            x="sector",
            y="position_value",
            title="Portfolio Value by Sector",
            labels={"position_value": "Value ($)", "sector": "Sector"},
            color="sector",
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    # Holdings detail table
    display_df = df[["ticker", "shares", "avg_cost", "current_price", "position_value", "gain_loss", "gain_loss_pct", "pe_ratio", "beta", "sector"]].copy()
    display_df.columns = ["Ticker", "Shares", "Avg Cost", "Current Price", "Value", "Gain/Loss ($)", "Gain/Loss %", "P/E", "Beta", "Sector"]
    for col in ["Avg Cost", "Current Price", "Value", "Gain/Loss ($)"]:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if x is not None else "—")
    display_df["Gain/Loss %"] = display_df["Gain/Loss %"].apply(lambda x: f"{x:+.2f}%")
    st.dataframe(display_df, use_container_width=True)


def _render_ai_analysis() -> None:
    st.subheader("AI Portfolio Analysis")

    if not st.session_state.get("portfolio_holdings"):
        st.info("Enter your holdings in the 'Holdings Input' tab first, then click 'Get AI Analysis'.")
        return

    if st.session_state.get("portfolio_ai_requested") or st.button("🤖 Analyze Portfolio", type="primary"):
        st.session_state.portfolio_ai_requested = False
        holdings = st.session_state.portfolio_holdings
        holdings_text = "\n".join(
            f"{h['ticker']}: {h['shares']} shares @ ${h['avg_cost']}"
            for h in holdings
        )
        prompt = f"Please analyze my investment portfolio:\n{holdings_text}"

        with st.spinner("AI is analyzing your portfolio..."):
            result = run_workflow(
                user_message=prompt,
                user_profile=st.session_state.get("user_profile"),
            )

        st.markdown(result["final_response"])
        label = result.get("agent_used", "")
        st.caption(f"Analysis by: Portfolio Analysis Agent")
