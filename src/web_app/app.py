"""Finnie Streamlit application — multi-tab financial assistant UI."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when running via `streamlit run`
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Bootstrap auth secrets from env vars (Cloud Run) before any Streamlit call
from src.web_app.auth_bootstrap import bootstrap_auth_secrets

bootstrap_auth_secrets()

import streamlit as st

from src.utils.logger import get_logger, setup_logging
from src.web_app.auth import (
    is_user_authorized,
    render_login_page,
    render_unauthorized_page,
    render_user_info_sidebar,
)

setup_logging()
logger = get_logger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Finnie — AI Finance Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Authentication gate ───────────────────────────────────────────────────────
if not st.user.is_logged_in:
    render_login_page()
    st.stop()

if not is_user_authorized():
    render_unauthorized_page()
    st.stop()

# ── Import page modules ───────────────────────────────────────────────────────
from src.web_app.pages.chat import render_chat_page
from src.web_app.pages.goals import render_goals_page
from src.web_app.pages.market import render_market_page
from src.web_app.pages.portfolio import render_portfolio_page

# ── Sidebar navigation ────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/savings.png", width=60)
        st.title("Finnie 💹")
        st.caption("AI-Powered Financial Education")
        st.divider()

        page = st.radio(
            "Navigate",
            options=["💬 Chat", "📊 Portfolio", "📈 Market", "🎯 Goals"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Your Profile")

        if "user_profile" not in st.session_state:
            st.session_state.user_profile = {
                "risk_tolerance": "moderate",
                "investment_horizon": "long",
                "knowledge_level": "beginner",
            }

        st.session_state.user_profile["knowledge_level"] = st.selectbox(
            "Knowledge Level",
            ["beginner", "intermediate", "advanced"],
            index=["beginner", "intermediate", "advanced"].index(
                st.session_state.user_profile.get("knowledge_level", "beginner")
            ),
        )
        st.session_state.user_profile["risk_tolerance"] = st.selectbox(
            "Risk Tolerance",
            ["conservative", "moderate", "aggressive"],
            index=["conservative", "moderate", "aggressive"].index(
                st.session_state.user_profile.get("risk_tolerance", "moderate")
            ),
        )
        st.session_state.user_profile["investment_horizon"] = st.selectbox(
            "Investment Horizon",
            ["short", "medium", "long"],
            index=["short", "medium", "long"].index(
                st.session_state.user_profile.get("investment_horizon", "long")
            ),
        )

        st.divider()
        st.caption("⚠️ Finnie provides financial education, not personalized advice. "
                   "Always consult a licensed financial advisor.")

        # Show logged-in user info at the bottom of the sidebar
        render_user_info_sidebar()

    return page


def main() -> None:
    page = render_sidebar()

    if page == "💬 Chat":
        render_chat_page()
    elif page == "📊 Portfolio":
        render_portfolio_page()
    elif page == "📈 Market":
        render_market_page()
    elif page == "🎯 Goals":
        render_goals_page()


if __name__ == "__main__":
    main()
