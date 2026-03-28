"""Chat page — multi-turn conversational interface."""
from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.workflow.graph import run_workflow

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

    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "lc_messages" not in st.session_state:
        st.session_state.lc_messages = []  # LangChain message objects for history

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
                st.caption(f"Handled by: {label}" + (f" — {reasoning}" if reasoning else ""))

    # Handle pending quick prompt
    if "pending_prompt" in st.session_state:
        user_input = st.session_state.pop("pending_prompt")
        _process_message(user_input)
        st.rerun()

    # Chat input
    if user_input := st.chat_input("Ask Finnie a financial question..."):
        _process_message(user_input)
        st.rerun()

    # Clear conversation
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.lc_messages = []
            st.rerun()


def _process_message(user_input: str) -> None:
    # Add user message to display history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show spinner while processing
    with st.spinner("Finnie is thinking..."):
        try:
            result = run_workflow(
                user_message=user_input,
                conversation_history=st.session_state.lc_messages,
                user_profile=st.session_state.get("user_profile"),
            )
            response = result["final_response"]
            agent_used = result["agent_used"]
            reasoning = result["router_reasoning"]

            # Update LangChain history for next turn
            st.session_state.lc_messages.append(HumanMessage(content=user_input))
            st.session_state.lc_messages.append(AIMessage(content=response))

            # Trim history to avoid context overflow
            max_history = 20
            if len(st.session_state.lc_messages) > max_history * 2:
                st.session_state.lc_messages = st.session_state.lc_messages[-(max_history * 2):]

            # Add assistant message to display history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "agent": agent_used,
                "reasoning": reasoning,
            })

        except Exception as exc:
            error_msg = f"Sorry, I encountered an error: {exc}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "agent": "error",
            })
