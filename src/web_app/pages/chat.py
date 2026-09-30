"""Chat page — multi-turn conversational interface.

The conversation is a persistent LangGraph thread owned by the signed-in user
(see UserData.active_thread_id), so it survives refreshes, restarts and a
second device. session_state only caches what is drawn on screen.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from src.core.config import get_settings
from src.core.llm import message_text
from src.web_app.session import current_user_data, persisted
from src.workflow.graph import (
    AGENT_NAMES,
    delete_conversation,
    load_conversation,
    run_workflow,
    stream_workflow,
)

_AGENT_LABELS = {
    "finance_qa": "📚 Finance Q&A",
    "portfolio": "📊 Portfolio Analysis",
    "market_analysis": "📈 Market Analysis",
    "goal_planning": "🎯 Goal Planning",
    "news_synthesizer": "📰 News Synthesis",
    "tax_education": "🧾 Tax Education",
    "error": "⚠️ Error",
}

_QUICK_PROMPTS = [
    "What is a P/E ratio?",
    "How does compound interest work?",
    "Explain the difference between Roth and Traditional IRA",
    "What is dollar-cost averaging?",
    "How should a beginner start investing?",
    "What are index funds?",
    "Explain tax-loss harvesting",
    "What's happening in the market today?",
]


def render_chat_page() -> None:
    st.title("💬 Chat with Finnie")
    st.caption("Ask anything about investing, markets, taxes, or financial planning")

    _ensure_conversation()

    # Quick prompts
    with st.expander("✨ Quick prompts", expanded=len(st.session_state.messages) == 0):
        cols = st.columns(4)
        for i, prompt in enumerate(_QUICK_PROMPTS):
            if cols[i % 4].button(prompt, key=f"qp_{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

    # Display conversation history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🧑" if role == "user" else "💹"):
            st.markdown(msg["content"])
            if role == "assistant" and "agent" in msg:
                label = _AGENT_LABELS.get(msg["agent"], msg["agent"])
                reasoning = msg.get("reasoning", "")
                prefix = "⚡ Instant answer — " if msg.get("cached") else "Handled by: "
                st.caption(f"{prefix}{label}" + (f" — {reasoning}" if reasoning else ""))

    # Handle pending quick prompt
    if "pending_prompt" in st.session_state:
        user_input = st.session_state.pop("pending_prompt")
        _process_message(user_input)
        st.rerun()

    # Chat input
    if user_input := st.chat_input("Ask Finnie a financial question..."):
        _process_message(user_input)
        st.rerun()

    # Clear conversation: delete the saved thread and start a new one.
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", type="secondary"):
            _clear_conversation()
            st.rerun()


def _ensure_conversation() -> None:
    """Resolve the user's chat thread and redraw it, once per browser session."""
    user_data = current_user_data()
    thread_id = st.session_state.get("chat_thread_id")
    # Re-resolve if the thread belongs to someone else (a sign-out and sign-in
    # within the same browser session), not just when there is none.
    if thread_id and user_data.owns_thread(thread_id):
        return

    thread_id = persisted(
        user_data.active_thread_id,
        None,
        "Couldn't load your saved conversation — this chat won't be saved.",
    )
    st.session_state.chat_thread_id = thread_id
    st.session_state.messages = (
        persisted(
            lambda: _display_messages(load_conversation(user_data.user_id, thread_id)),
            [],
            "Couldn't load your earlier messages.",
        )
        if thread_id
        else []
    )


def _clear_conversation() -> None:
    user_data = current_user_data()
    old_thread: Optional[str] = st.session_state.get("chat_thread_id")
    if old_thread:
        persisted(
            lambda: delete_conversation(user_data.user_id, old_thread),
            None,
            "Couldn't delete the saved conversation.",
        )
    st.session_state.chat_thread_id = persisted(
        user_data.new_thread, None, "Couldn't start a new saved conversation."
    )
    st.session_state.messages = []


def _display_messages(messages: list) -> list[dict]:
    """Chat bubbles for a saved thread.

    Routing reasoning and cache-hit flags are not stored, so a redrawn answer
    shows only which agent handled it.
    """
    shown: list[dict] = []
    for message in messages:
        if message.type == "human":
            shown.append({"role": "user", "content": message_text(message)})
        elif message.type == "ai":
            entry = {"role": "assistant", "content": message_text(message)}
            # Agents sign with their display name; FAQ-cache hits with the node key.
            agent = AGENT_NAMES.get(message.name or "", message.name)
            if agent in _AGENT_LABELS:
                entry["agent"] = agent
            shown.append(entry)
    return shown


def _run_turn(user_input: str) -> dict:
    """Execute one workflow turn, streaming into the chat when enabled.

    The history comes from the saved thread. If persistence was unavailable
    (no thread id), the turn runs one-shot, without earlier context.
    """
    turn = {
        "user_message": user_input,
        "user_profile": st.session_state.get("user_profile"),
        "user_id": current_user_data().user_id,
        "thread_id": st.session_state.get("chat_thread_id"),
    }
    if not get_settings().fast_path.streaming:
        with st.spinner("Finnie is thinking..."):
            return run_workflow(**turn)

    # Streaming path: render tokens as they arrive; `sink` receives the full
    # result once the graph finishes.
    sink: dict = {}
    with st.chat_message("assistant", avatar="💹"):
        st.write_stream(stream_workflow(**turn, sink=sink))
    return sink


def _process_message(user_input: str) -> None:
    # Add user message to display history
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        result = _run_turn(user_input)
        response = result.get("final_response", "")
        agent_used = result.get("agent_used", "finance_qa")
        reasoning = result.get("router_reasoning", "")

        # Add assistant message to display history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "agent": agent_used,
            "reasoning": reasoning,
            "cached": bool(result.get("cache_hit", False)),
        })

    except Exception as exc:
        error_msg = f"Sorry, I encountered an error: {exc}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "agent": "error",
        })
