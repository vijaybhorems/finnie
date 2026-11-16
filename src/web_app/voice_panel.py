"""Sidebar panel: voice settings and past calls (design doc, Page changes item 5).

Built like the memory panel: the saved "Speak amounts by default" setting,
and the user's past voice sessions, each deletable. Deleting removes the
transcript (the call's thread, through delete_conversation() and its
ownership check) and the call's record. Past calls can't be reopened yet.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.persistence.user_data import UserData
from src.web_app.session import persisted


def _when(call: dict) -> str:
    try:
        started = datetime.fromisoformat(call["started_at"]).astimezone()
    except (KeyError, ValueError):
        return "Voice call"
    return started.strftime("%d %b, %H:%M")


def _length(call: dict, active: bool) -> str:
    if active:
        return "in progress"
    seconds = call.get("seconds")
    if seconds is None:  # the worker stopped before it saved the call's length
        return "length unknown"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


def render_voice_panel(user_data: UserData, active_room: str | None) -> None:
    with st.expander("Voice calls", icon=":material/mic:", expanded=False):
        speak = persisted(user_data.speak_amounts, False, "Couldn't load your voice setting.")
        wanted = st.toggle(
            "Speak amounts by default",
            value=speak,
            key="speak_amounts_default_toggle",
            help="When off, Finnie keeps dollar amounts on your screen instead of saying them. "
                 "You can change it for one call with the switch on the call bar.",
        )
        if wanted != speak:
            persisted(lambda: user_data.set_speak_amounts(wanted), None, "Couldn't save your voice setting.")

        calls = persisted(lambda: user_data.voice_calls(limit=20), [], "Couldn't load your voice calls.")
        if not calls:
            st.caption("No calls yet. Press Talk beside the chat box to start one.")
            return
        st.caption("Transcripts can be deleted. Reopening a past call isn't available yet.")
        with st.container(key="fn-voice-calls"):
            for call in calls:
                thread_id = call.get("thread_id", "")
                active = thread_id == active_room
                text_col, delete_col = st.columns([5, 1], wrap=False)
                text_col.markdown(f"{_when(call)}  \n:gray[{_length(call, active)}]")
                if active:
                    delete_col.caption("now")
                    continue
                if delete_col.button("Delete this call", icon=":material/delete:", key=f"del_call_{thread_id}",
                                     help="Delete this call's transcript", type="tertiary"):
                    persisted(lambda t=thread_id: _delete_call(user_data, t), None,
                              "Couldn't delete that call.")
                    st.rerun()


def _delete_call(user_data: UserData, thread_id: str) -> None:
    from src.workflow.graph import delete_conversation

    delete_conversation(user_data.user_id, thread_id)
    user_data.delete_voice_call(thread_id)
