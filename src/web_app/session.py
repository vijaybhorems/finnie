"""Per-request access to the signed-in user's saved data from Streamlit pages."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

import streamlit as st

from src.persistence.identity import user_id_from_claims
from src.persistence.user_data import UserData
from src.utils.logger import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def current_user_id() -> str:
    """The signed-in user's id, derived from the verified OIDC claims.

    Recomputed on every call rather than cached in session_state, so identity
    can never outlive a sign-out/sign-in within the same browser session.
    """
    return user_id_from_claims(getattr(st.user, "sub", None), getattr(st.user, "email", None))


def current_user_data() -> UserData:
    return UserData(current_user_id())


def persisted(action: Callable[[], T], default: T, failure_message: str) -> T:
    """Run a persistence call; on failure warn in the page and return `default`.

    A database hiccup should cost the user their saved state for one page view,
    not the whole page.
    """
    try:
        return action()
    except Exception as exc:  # noqa: BLE001
        logger.error("persistence_call_failed", error=str(exc), error_type=type(exc).__name__)
        st.warning(failure_message)
        return default


def save_if_changed(state_key: str, value: Any, save: Callable[[Any], Any], failure_message: str) -> None:
    """Save `value` once it differs from the baseline held in session_state[state_key].

    The baseline is whatever the page first showed — saved data, or a default
    or demo value. Only a real change is written, so a user who never touches
    a demo value never has it saved as their own.
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = value
        return
    if value == st.session_state[state_key]:
        return
    persisted(lambda: save(value), None, failure_message)
    st.session_state[state_key] = value
