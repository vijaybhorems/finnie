"""Sidebar panel: what Finnie remembers about the user, with delete and an off switch."""
from __future__ import annotations

import streamlit as st

from src.persistence.user_data import UserData
from src.web_app.markdown import render_safe
from src.web_app.session import persisted


def render_memory_panel(user_data: UserData) -> None:
    with st.expander("🧠 What Finnie remembers", expanded=False):
        enabled = persisted(user_data.memory_enabled, True, "Couldn't load your memory setting.")
        wanted = st.toggle(
            "Remember things I share",
            value=enabled,
            key="memory_enabled_toggle",
            help="When on, Finnie saves lasting facts you mention in chat (goals, "
                 "timeline, situation) and uses them in later conversations. When "
                 "off, it neither saves nor uses them; saved facts stay until you delete them.",
        )
        if wanted != enabled:
            persisted(lambda: user_data.set_memory_enabled(wanted), None,
                      "Couldn't save your memory setting.")

        memories = persisted(user_data.list_memories, [], "Couldn't load your saved memories.")
        if not memories:
            st.caption("Nothing saved yet. Mention a goal or your situation in chat and it will appear here.")
            return

        for memory in memories:
            text_col, delete_col = st.columns([6, 1])
            text_col.markdown(
                f"{render_safe(memory['fact'])}  \n"
                f":gray[{memory.get('category', '')} · noted {memory.get('noted_on', '')}]"
            )
            if delete_col.button("🗑️", key=f"forget_{memory['id']}", help="Forget this"):
                persisted(lambda m=memory: user_data.forget(m["id"]), None, "Couldn't delete that memory.")
                st.rerun()

        if st.button("Forget everything", key="forget_all_memories"):
            persisted(user_data.forget_all, 0, "Couldn't delete your memories.")
            st.rerun()
