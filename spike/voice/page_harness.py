"""The real Chat page with voice, for local testing without Google sign-in.

Renders src/web_app's Chat page and Voice calls panel; only the signed-in
identity is replaced by a fixed test user (g-voicepage00000000001, email
tester@example.com). Local only: spike/ is not in the Docker image.

    PYTHONPATH=. .venv/bin/streamlit run spike/voice/page_harness.py --server.port 8530

Needs the LiveKit dev server and the worker (README, "Run it locally"), started
with FINNIE_VOICE_ALLOWED_EMAILS=tester@example.com. Point both at the same
DATABASE_URL: without one, each process has its own memory store, so calls the
worker records never reach the sidebar.
"""
import os

os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")
os.environ["FINNIE_VOICE_ENABLED"] = "1"
os.environ.setdefault("FINNIE_VOICE_ALLOWED_EMAILS", "tester@example.com")

import streamlit as st  # noqa: E402

import src.web_app.session as session  # noqa: E402
import src.web_app.voice as voice  # noqa: E402

EMAIL = "tester@example.com"
session.user_id_from_claims = lambda *_: "g-voicepage00000000001"
voice._email = lambda: EMAIL

from src.web_app.theme import apply_theme  # noqa: E402
from src.web_app.views.chat import render_chat_page  # noqa: E402
from src.web_app.voice_panel import render_voice_panel  # noqa: E402

st.set_page_config(page_title="Finnie voice page", layout="wide")
apply_theme()
with st.sidebar:
    render_voice_panel(session.current_user_data(), voice.active_call_room())
render_chat_page()
