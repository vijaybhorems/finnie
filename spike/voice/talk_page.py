"""Spike Talk page: proves the microphone works inside a Streamlit component.

LOCAL USE ONLY. No sign-in: it mints LiveKit tokens for whoever opens it, so
it must never be deployed (spike/ is excluded from the Docker image). The real
button sits on the chat page behind Google sign-in (design doc, Page changes).

Run (from the repo root, so `spike` and `src` import):
    PYTHONPATH=. .venv-voice/bin/streamlit run spike/voice/talk_page.py
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from spike.voice.local_token import LIVEKIT_URL, TEST_EMAIL, new_call

# Pinned, as the design doc asks of every CDN script.
LIVEKIT_CLIENT_JS = "https://cdn.jsdelivr.net/npm/livekit-client@2.23.0/dist/livekit-client.umd.min.js"


st.set_page_config(page_title="Finnie voice spike", page_icon="🎙️")
st.title("Voice spike: Talk")
st.caption("Local test page. Start the LiveKit dev server and the worker first (spike/voice/README.md).")

name = st.text_input("Test user", value="tester")
email = st.text_input("Email in the token (must be on the worker's allowlist)", value=TEST_EMAIL)
if st.session_state.get("call_for") != (name, email):
    st.session_state.call_for = (name, email)
    st.session_state.call = new_call(name, email)
identity, room, token = st.session_state.call
st.caption(f"Identity `{identity}`, room `{room}`")
config = json.dumps({"url": LIVEKIT_URL, "token": token, "room": room})

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
