"""Authentication helpers — Google OIDC via Streamlit's built-in auth."""
from __future__ import annotations

import html
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
    """Render the branded sign-in page for unauthenticated users."""
    from src.web_app.theme import render_login_page as render_page

    render_page(lambda: st.login("google"))


def render_unauthorized_page() -> None:
    """Render a page for authenticated but unauthorized users."""
    st.error("⛔ Your account is not authorized to access this application.")
    st.info(f"Signed in as: **{st.user.email}**")
    st.caption("Contact the administrator to request access.")
    if st.button("Sign out"):
        st.logout()


def render_user_info_sidebar() -> None:
    """Show the logged-in user's info and a sign-out button in the sidebar."""
    name = getattr(st.user, "name", None) or st.user.email
    avatar = getattr(st.user, "picture", None)
    face = (
        f'<img src="{html.escape(avatar, quote=True)}" alt="" referrerpolicy="no-referrer">'
        if avatar
        else f'<div class="fn-avatar">{html.escape(name[:1].upper())}</div>'
    )
    with st.sidebar:
        st.divider()
        st.markdown(
            f'<div class="fn-user">{face}<div style="min-width:0">'
            f'<div class="fn-user-name">{html.escape(name)}</div>'
            f'<div class="fn-user-mail">{html.escape(st.user.email)}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Sign out", key="sidebar_logout", use_container_width=True,
                     icon=":material/logout:"):
            st.logout()
