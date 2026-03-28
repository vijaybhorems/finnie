"""Market overview page — live prices, index charts, sector heatmap."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.alpha_vantage_client import AlphaVantageClient
from src.data.yfinance_client import YFinanceClient
from src.workflow.graph import run_workflow

_MAJOR_INDICES = {
    "S&P 500": "SPY",
    "NASDAQ 100": "QQQ",
    "Dow Jones": "DIA",
    "Russell 2000": "IWM",
    "Gold": "GLD",
    "Bonds (AGG)": "AGG",
}


def render_market_page() -> None:
    st.title("📈 Market Overview")
    st.caption("Real-time market data, sector performance, and AI market analysis")

    tab1, tab2, tab3 = st.tabs(["🌍 Market Snapshot", "📉 Price Charts", "🤖 AI Market Analysis"])

    with tab1:
        _render_market_snapshot()
    with tab2:
        _render_price_charts()
    with tab3:
        _render_ai_market_analysis()


def _render_market_snapshot() -> None:
    yf_client = YFinanceClient()
    av_client = AlphaVantageClient()

    st.subheader("Major Indices")
    with st.spinner("Loading market data..."):
        cols = st.columns(len(_MAJOR_INDICES))
        for i, (name, ticker) in enumerate(_MAJOR_INDICES.items()):
            data = yf_client.get_current_price(ticker)
            price = data.get("current_price")
            change_pct = data.get("change_pct")
            if price and change_pct is not None:
                cols[i].metric(
                    name,
                    f"${price:,.2f}",
                    f"{change_pct:+.2f}%",
                    delta_color="normal",
                )
            else:
                cols[i].metric(name, "—", "—")

    # Sector performance heatmap
    st.subheader("Sector Performance (1-Day)")
    with st.spinner("Loading sector data..."):
        sector_data = av_client.get_sector_performance()
        if sector_data and "error" not in sector_data:
            one_day = sector_data.get("one_day", {})
            if one_day:
                sectors = list(one_day.keys())
                values = [float(v.strip("%")) for v in one_day.values()]
                df = pd.DataFrame({"Sector": sectors, "Return": values})
                df = df.sort_values("Return", ascending=True)

                colors = ["#ef5350" if x < 0 else "#26a69a" for x in df["Return"]]
                fig = go.Figure(go.Bar(
                    x=df["Return"],
                    y=df["Sector"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.2f}%" for v in df["Return"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="US Sector Performance (1-Day)",
                    xaxis_title="Return %",
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sector data requires an Alpha Vantage API key. Configure it in .env to see sector heatmaps.")

    # Watchlist
    st.subheader("Custom Watchlist")
    watchlist_input = st.text_input(
        "Enter tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA",
        key="watchlist_input",
    )
    if st.button("Refresh Watchlist", type="secondary"):
        tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
        with st.spinner("Fetching watchlist..."):
            rows = []
            for ticker in tickers[:10]:
                data = yf_client.get_current_price(ticker)
                rows.append({
                    "Ticker": ticker,
                    "Price": f"${data.get('current_price', 0):,.2f}" if data.get('current_price') else "—",
                    "Change %": f"{data.get('change_pct', 0):+.2f}%" if data.get('change_pct') is not None else "—",
                    "Day High": f"${data.get('day_high', 0):,.2f}" if data.get('day_high') else "—",
                    "Day Low": f"${data.get('day_low', 0):,.2f}" if data.get('day_low') else "—",
                    "Market Cap": _format_market_cap(data.get('market_cap')),
                })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)


def _render_price_charts() -> None:
    st.subheader("Historical Price Chart")
    yf_client = YFinanceClient()

    col1, col2 = st.columns(2)
    ticker = col1.text_input("Ticker Symbol", value="SPY", key="chart_ticker").upper()
    period = col2.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="chart_period")

    if st.button("📈 Load Chart", type="primary"):
        with st.spinner(f"Loading {ticker} history..."):
            history = yf_client.get_historical_prices(ticker, period=period)

        if not history:
            st.error(f"No data found for {ticker}")
            return

        df = pd.DataFrame(history)

        # Identify the closing price column
        close_col = None
        for col in df.columns:
            if "close" in col.lower():
                close_col = col
                break

        date_col = None
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break

        if close_col and date_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[close_col],
                mode="lines",
                name=ticker,
                line=dict(color="#2196F3", width=2),
                fill="tozeroy",
                fillcolor="rgba(33, 150, 243, 0.1)",
            ))
            fig.update_layout(
                title=f"{ticker} — {period} Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode="x unified",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Volume chart
            vol_col = next((c for c in df.columns if "volume" in c.lower()), None)
            if vol_col:
                fig_vol = px.bar(df, x=date_col, y=vol_col, title=f"{ticker} Volume", color_discrete_sequence=["#78909C"])
                fig_vol.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.warning("Could not parse price data from response.")


def _render_ai_market_analysis() -> None:
    st.subheader("AI Market Analysis")

    quick_queries = [
        "What's the current state of the stock market?",
        "Explain today's sector performance",
        "What are the key market trends this week?",
        "Is the market overbought or oversold?",
    ]

    selected = st.selectbox("Quick Analysis Topics", [""] + quick_queries, key="market_quick")
    custom = st.text_input("Or ask a custom question", key="market_custom")

    query = custom if custom else selected
    if query and st.button("🤖 Get Analysis", type="primary", key="market_analyze"):
        with st.spinner("Analyzing market conditions..."):
            result = run_workflow(
                user_message=query,
                user_profile=st.session_state.get("user_profile"),
            )
        st.markdown(result["final_response"])


def _format_market_cap(cap) -> str:
    if not cap:
        return "—"
    if cap >= 1e12:
        return f"${cap/1e12:.1f}T"
    if cap >= 1e9:
        return f"${cap/1e9:.1f}B"
    if cap >= 1e6:
        return f"${cap/1e6:.1f}M"
    return f"${cap:,.0f}"
