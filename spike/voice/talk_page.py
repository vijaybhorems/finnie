"""Spike Talk page: proves the microphone works inside a Streamlit component.

LOCAL USE ONLY. No sign-in: it mints LiveKit tokens for whoever opens it, so
it must never be deployed (spike/ is excluded from the Docker image). The real
button sits on the chat page behind Google sign-in (design doc, Page changes).

Run:
    .venv-voice/bin/streamlit run spike/voice/talk_page.py
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import timedelta

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from livekit import api

load_dotenv()

# Defaults match `livekit-server --dev` (see README); LiveKit Cloud values come from .env.
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
# Pinned, as the design doc asks of every CDN script.
LIVEKIT_CLIENT_JS = "https://cdn.jsdelivr.net/npm/livekit-client@2.23.0/dist/livekit-client.umd.min.js"


def mint_token(identity: str, room: str) -> str:
    """Join one room, publish microphone audio only (design doc: token rules)."""
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=False,
        can_publish_sources=["microphone"],
    )
    return (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(grants)
        .with_ttl(timedelta(minutes=10))
        .to_jwt()
    )


st.set_page_config(page_title="Finnie voice spike", page_icon="🎙️")
st.title("Voice spike: Talk")
st.caption("Local test page. Start the LiveKit dev server and the worker first (spike/voice/README.md).")

identity = st.text_input("Test identity", value="tester")
if "room" not in st.session_state:
    st.session_state.room = f"spike-{identity}:{uuid.uuid4()}"
config = json.dumps({
    "url": LIVEKIT_URL,
    "token": mint_token(identity, st.session_state.room),
    "room": st.session_state.room,
})

components.html(
    f"""
<div style="font-family: sans-serif">
  <button id="talk" style="font-size:16px;padding:8px 18px;border-radius:10px;border:1.5px solid #0E7C66;color:#0E7C66;background:#fff;cursor:pointer">🎙️ Talk</button>
  <button id="end" style="display:none;font-size:16px;padding:8px 18px;border-radius:10px;border:1px solid #c0392b;color:#c0392b;background:#fff;cursor:pointer">End</button>
  <span id="status" style="margin-left:12px;color:#555">Idle</span>
  <div id="log" style="margin-top:12px;font-size:14px;line-height:1.5;max-height:340px;overflow:auto"></div>
</div>
<script src="{LIVEKIT_CLIENT_JS}"></script>
<script>
const cfg = {config};
const LK = window.LivekitClient;
const $ = (id) => document.getElementById(id);
let room = null;

function line(who, text) {{
  const d = document.createElement("div");
  d.innerHTML = "<b>" + who + ":</b> " + text.replace(/</g, "&lt;");
  $("log").appendChild(d); $("log").scrollTop = 1e9;
}}

$("talk").onclick = async () => {{
  $("status").textContent = "Connecting…";
  try {{
    room = new LK.Room({{ audioCaptureDefaults: {{ echoCancellation: true, noiseSuppression: true, autoGainControl: true }} }});
    room.on(LK.RoomEvent.TrackSubscribed, (track) => {{
      if (track.kind === "audio") document.body.appendChild(track.attach());
    }});
    // Agents 1.x send transcripts as "lk.transcription" text streams (the old event is deprecated).
    room.registerTextStreamHandler("lk.transcription", async (reader, info) => {{
      const text = await reader.readAll();
      if (text.trim()) line(info.identity === room.localParticipant.identity ? "You" : "Finnie", text);
    }});
    room.on(LK.RoomEvent.ParticipantAttributesChanged, (changed, p) => {{
      if (changed["lk.agent.state"]) $("status").textContent = changed["lk.agent.state"];
    }});
    room.on(LK.RoomEvent.Disconnected, () => {{ $("status").textContent = "Call ended"; $("talk").style.display = ""; $("end").style.display = "none"; }});
    await room.connect(cfg.url, cfg.token);
    await room.localParticipant.setMicrophoneEnabled(true);
    $("status").textContent = "Connected — waiting for Finnie";
    $("talk").style.display = "none"; $("end").style.display = "";
  }} catch (e) {{
    $("status").textContent = "Failed: " + e.message;
    line("Error", String(e));
  }}
}};
$("end").onclick = () => room && room.disconnect();
</script>
""",
    height=440,
)

with st.expander("What this checks"):
    st.markdown(
        "- The browser's microphone prompt appears inside the component frame and audio reaches the worker.\n"
        "- Echo cancellation: on laptop speakers, Finnie must not hear and answer itself.\n"
        "- iOS Safari: open this page from the phone (same network, `--server.address 0.0.0.0`; "
        "iOS needs HTTPS for the microphone, e.g. a tunnel), then lock the screen mid-answer."
    )
