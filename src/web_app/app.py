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
from src.web_app import warmup
from src.web_app.auth import (
    is_user_authorized,
    render_login_page,
    render_unauthorized_page,
    render_user_info_sidebar,
)
from src.web_app.theme import apply_theme, render_brand

setup_logging()
logger = get_logger(__name__)

# Load LangGraph + the embedding model in a background thread (a no-op if
# src.web_app.serve already started it at boot). Nothing above this line, and
# nothing the sign-in page draws, imports LangChain — so the sign-in page paints
# immediately on a cold instance while the ~minute of imports runs alongside,
# overlapping the user's Google round trip. Tracing is registered inside the
# warm-up, before it imports LangChain.
warmup.start()

# ── Page config (must be first Streamlit call) ────────────────────────────────
# The Docker image writes the same title and icon into Streamlit's index.html,
# so the tab reads "Finnie" before this runs too (docker/Dockerfile).
st.set_page_config(
    page_title="Finnie — AI Finance Assistant",
    page_icon=str(Path(__file__).with_name("assets") / "favicon.png"),
    layout="wide",
    # Open on desktop, collapsed behind the ☰ button on phones.
    initial_sidebar_state="auto",
)
apply_theme()

# ── Authentication gate ───────────────────────────────────────────────────────
if not st.user.is_logged_in:
    render_login_page()
    st.stop()

if not is_user_authorized():
    render_unauthorized_page()
    st.stop()

# Persistence imports LangGraph; keep them behind the gate (see warm-up above),
# and after tracing is registered so the instrumentor sees LangChain first.
warmup.wait_for_tracing()
from src.persistence.user_data import DEFAULT_PROFILE  # noqa: E402
from src.web_app.memory_panel import render_memory_panel  # noqa: E402
from src.web_app.session import current_user_data, persisted, save_if_changed  # noqa: E402

# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = ["💬\u2002Chat", "📊\u2002Portfolio", "📈\u2002Market", "🎯\u2002Goals"]

# Two widgets pick the page: the sidebar radio (desktop) and a tab bar at the
# top of the page that only phones see (theme.py hides it on wide screens,
# where the sidebar is always open). "nav" is the source of truth; each widget
# copies its choice into it and the other widget's key follows.


def _nav_from_sidebar() -> None:
    st.session_state.nav_mobile = st.session_state.nav


def _nav_from_mobile() -> None:
    # A segmented control can be clicked off to nothing; keep the current page.
    if st.session_state.nav_mobile is None:
        st.session_state.nav_mobile = st.session_state.nav
    else:
        st.session_state.nav = st.session_state.nav_mobile


def render_mobile_nav() -> None:
    st.session_state.setdefault("nav_mobile", st.session_state.nav)
    with st.container(key="fn-mobile-nav"):
        st.segmented_control(
            "Navigate",
            PAGES,
            key="nav_mobile",
            on_change=_nav_from_mobile,
            label_visibility="collapsed",
        )


def render_sidebar() -> str:
    st.session_state.setdefault("nav", PAGES[0])
    with st.sidebar:
        render_brand()

        page = st.radio(
            "Navigate",
            options=PAGES,
            key="nav",
            on_change=_nav_from_sidebar,
            label_visibility="collapsed",
        )

        st.divider()

        # The profile is saved per user, so it follows them across refreshes
        # and devices, and the agents read it from the store (hydrate node).
        user_data = current_user_data()
        if "user_profile" not in st.session_state:
            st.session_state.user_profile = persisted(
                user_data.get_profile,
                dict(DEFAULT_PROFILE),
                "Couldn't load your saved profile — showing defaults.",
            )

        profile_box = st.expander("Your profile", icon=":material/tune:", expanded=False)
        profile_box.caption("Finnie tailors explanations to these.")
        st.session_state.user_profile["knowledge_level"] = profile_box.selectbox(
            "Knowledge Level",
            ["beginner", "intermediate", "advanced"],
            index=["beginner", "intermediate", "advanced"].index(
                st.session_state.user_profile.get("knowledge_level", "beginner")
            ),
        )
        st.session_state.user_profile["risk_tolerance"] = profile_box.selectbox(
            "Risk Tolerance",
            ["conservative", "moderate", "aggressive"],
            index=["conservative", "moderate", "aggressive"].index(
                st.session_state.user_profile.get("risk_tolerance", "moderate")
            ),
        )
        st.session_state.user_profile["investment_horizon"] = profile_box.selectbox(
            "Investment Horizon",
            ["short", "medium", "long"],
            index=["short", "medium", "long"].index(
                st.session_state.user_profile.get("investment_horizon", "long")
            ),
        )

        # A copy: the selectboxes mutate user_profile in place, and a baseline
        # that is the same dict would change with it and never look different.
        save_if_changed(
            "_saved_profile",
            dict(st.session_state.user_profile),
            user_data.save_profile,
            "Couldn't save your profile right now — it will reset on refresh.",
        )

        render_memory_panel(user_data)

        st.markdown(
            '<div class="fn-fineprint">Finnie provides financial education, not '
            "personalised advice. Consult a licensed advisor before acting.</div>",
            unsafe_allow_html=True,
        )

        # Show logged-in user info at the bottom of the sidebar
        render_user_info_sidebar()

    return page


def main() -> None:
    page = render_sidebar()
    render_mobile_nav()

    # Every tab runs the workflow, so wait for the warm-up here. Usually it has
    # long finished by the time sign-in completes; on a cold instance this is
    # the only wait, and the sidebar is already drawn around it.
    if not warmup.is_ready():
        with st.spinner("Finnie is getting ready — this only happens after a quiet spell…"):
            warmup.wait()

    # Import page modules lazily so landing on the default Chat tab doesn't pull
    # in the other tabs' dependencies (plotly, yfinance, portfolio/market/goals
    # code). Each module is imported once per process, then cached in sys.modules.
    if page == "💬\u2002Chat":
        from src.web_app.views.chat import render_chat_page
        render_chat_page()
    elif page == "📊\u2002Portfolio":
        from src.web_app.views.portfolio import render_portfolio_page
        render_portfolio_page()
    elif page == "📈\u2002Market":
        from src.web_app.views.market import render_market_page
        render_market_page()
    elif page == "🎯\u2002Goals":
        from src.web_app.views.goals import render_goals_page
        render_goals_page()


if __name__ == "__main__":
    main()
