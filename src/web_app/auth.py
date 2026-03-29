"""Authentication helpers — Google OIDC via Streamlit's built-in auth."""
from __future__ import annotations

import os

import streamlit as st

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_allowed_emails() -> set[str]:
    """Load allowed emails from environment variable.

    Format: comma-separated list, e.g. 'alice@gmail.com,bob@company.com'
    If empty or unset, all authenticated Google users are allowed.
    """
    raw = os.environ.get("ALLOWED_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_user_authorized() -> bool:
    """Check if the current user is authenticated and on the allowlist."""
    if not st.user.is_logged_in:
        return False
    allowed = get_allowed_emails()
    if not allowed:
        # No allowlist configured → allow all authenticated users
        return True
    return st.user.email.lower() in allowed


def render_login_page() -> None:
    """Render a branded login page for unauthenticated users."""
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 0;">
            <h1>💹 Finnie</h1>
            <h3>AI-Powered Financial Education</h3>
            <p style="color: grey;">Sign in with your Google account to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔐 Sign in with Google", use_container_width=True):
            st.login("google")


def render_unauthorized_page() -> None:
    """Render a page for authenticated but unauthorized users."""
    st.error("⛔ Your account is not authorized to access this application.")
    st.info(f"Signed in as: **{st.user.email}**")
    st.caption("Contact the administrator to request access.")
    if st.button("Sign out"):
        st.logout()


def render_user_info_sidebar() -> None:
    """Show the logged-in user's info and a sign-out button in the sidebar."""
    with st.sidebar:
        st.divider()
        cols = st.columns([1, 3])
        with cols[0]:
            avatar = getattr(st.user, "picture", None)
            if avatar:
                st.image(avatar, width=40)
            else:
                st.markdown("👤")
        with cols[1]:
            name = getattr(st.user, "name", st.user.email)
            st.markdown(f"**{name}**")
            st.caption(st.user.email)
        if st.button("Sign out", key="sidebar_logout", use_container_width=True):
            st.logout()
