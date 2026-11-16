"""Voice on the Chat page: the Talk button, consent, and the call itself.

Design doc, Page changes. Shown only to users voice is allowed for
(`voice.enabled` and the allowlist); for everyone else the page is unchanged.

- Talk sits beside the chat input. The first press shows the privacy notice;
  after that it mints a LiveKit token on the server (src/voice/tokens.py) for
  the signed-in identity and starts the call.
- During a call the conversation steps aside for the voice session view, an
  st.components.v2 component (voice_call/) that holds the LiveKit connection
  and draws the live transcript, full answers and call bar itself, so no line
  causes a Streamlit rerun. Its "Speak amounts" switch saves this call's
  setting; `ended` brings the typed conversation back.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from src.core.config import get_settings
from src.utils.logger import get_logger
from src.voice.access import is_allowed
from src.web_app.session import current_user_data, current_user_id, persisted

logger = get_logger(__name__)

_ASSETS = Path(__file__).parent / "voice_call"
_CALL_KEY = "voice_call"

CONSENT_TEXT = """\
**Before your first call**

- **Where your voice goes:** your microphone audio travels through **LiveKit**, which carries the call, \
to **Deepgram**, which turns it into text. Finnie's answers are spoken by **ElevenLabs**, or by Deepgram \
if ElevenLabs is unavailable.
- **What's saved:** the call's transcript is saved as a voice session. You can delete it from \
**Voice calls** in the sidebar.
- **Amounts stay on screen:** Finnie doesn't say dollar amounts aloud unless you turn on \
**Speak amounts**. The full answers, figures included, appear on your screen.
- As in chat, Finnie may remember lasting facts you mention, such as a goal. You can see and delete \
them under **What Finnie remembers**.
"""


# Calls that failed, by the reason the component gives (voice_call/call.js).
END_MESSAGES = {
    "mic_denied": "Finnie needs your microphone. Allow it for this site in your browser, then press Talk again.",
    "load_failed": "The call couldn't load. Check your connection and try again.",
    "failed": "The call couldn't start. Try again in a moment, or keep typing.",
    "already_ended": "That call has ended. Press Talk to start a new one.",
    "no_agent": "Finnie couldn't join the call. Try again in a moment, or keep typing.",
    "agent_left": "The call dropped. Press Talk to start a new one.",
}


def _email() -> str:
    """The signed-in user's verified email (Google OIDC claims)."""
    return getattr(st.user, "email", None) or ""


def voice_offered() -> bool:
    """Whether this signed-in user sees voice at all."""
    return is_allowed(_email(), get_settings().voice)


def in_call() -> bool:
    return bool(st.session_state.get(_CALL_KEY))


def active_call_room() -> Optional[str]:
    call = st.session_state.get(_CALL_KEY)
    return call["room"] if call else None


def render_talk_button(container: Any) -> None:
    """The Talk button, in `container` (beside the chat input)."""
    if container.button("Talk", icon=":material/mic:", key="voice_talk", help="Talk to Finnie",
                        width="content"):
        if persisted(current_user_data().voice_consented, False, "Couldn't load your voice setting."):
            if _start_call():
                st.rerun()  # the page was drawn before the call existed
        else:
            _consent_dialog()


@st.dialog("Talk to Finnie")
def _consent_dialog() -> None:
    st.markdown(CONSENT_TEXT)
    start, later = st.columns(2)
    if start.button("Start call", type="primary", icon=":material/mic:", use_container_width=True):
        persisted(current_user_data().set_voice_consent, None, "Couldn't save your choice.")
        if _start_call():
            st.rerun()
    if later.button("Not now", use_container_width=True):
        st.rerun()


def _start_call() -> bool:
    """Mint a token for a new call and switch the page to it. False if refused."""
    from src.voice.tokens import VoiceRefused, mint_call_token

    try:
        token = mint_call_token(current_user_id(), _email())
    except VoiceRefused as refused:
        st.warning(refused.message, icon=":material/mic_off:")
        return False
    except Exception as exc:  # noqa: BLE001 — a failed start must not break the chat page
        logger.error("voice_call_start_failed", error_type=type(exc).__name__)
        st.warning("Voice isn't available right now. You can keep typing.", icon=":material/mic_off:")
        return False
    st.session_state[_CALL_KEY] = {**asdict(token), "started": time.time()}
    return True


@st.cache_resource
def _component():
    return st.components.v2.component(
        "finnie_voice_call",
        html=(_ASSETS / "call.html").read_text(),
        css=(_ASSETS / "call.css").read_text(),
        js=(_ASSETS / "call.js").read_text(),
    )


def render_voice_session(agent_labels: dict[str, str]) -> None:
    """The call in place of the conversation, until it ends."""
    call = st.session_state[_CALL_KEY]
    room = call["room"]
    user_data = current_user_data()
    if "voice_speak_amounts" not in st.session_state or st.session_state.get("voice_speak_room") != room:
        st.session_state.voice_speak_amounts = persisted(
            lambda: user_data.speak_amounts(room), False, "Couldn't load your Speak amounts setting.",
        )
        st.session_state.voice_speak_room = room

    result = _component()(
        key=f"fv_{room.rsplit(':', 1)[-1]}",
        data={
            "url": call["url"],
            "token": call["token"],
            "room": room,
            "speak_amounts": st.session_state.voice_speak_amounts,
            "labels": agent_labels,
        },
        on_ended_change=lambda: None,
        on_speak_amounts_change=lambda: None,
    )

    wanted = getattr(result, "speak_amounts", None)
    if isinstance(wanted, bool) and wanted != st.session_state.voice_speak_amounts:
        persisted(lambda: user_data.set_speak_amounts(wanted, room), None,
                  "Couldn't save Speak amounts for this call.")
        st.session_state.voice_speak_amounts = wanted

    if getattr(result, "ended", None):
        end_call(str(result.ended))
        st.rerun()


def end_call(reason: str) -> None:
    """Back to the typed conversation."""
    call = st.session_state.pop(_CALL_KEY, None)
    if not call:
        return
    minutes, seconds = divmod(int(time.time() - call["started"]), 60)
    logger.info("voice_call_left_page", reason=reason)
    if reason in END_MESSAGES:
        st.toast(END_MESSAGES[reason], icon=":material/mic_off:")
    else:
        st.toast(f"Call ended after {minutes}:{seconds:02d}. Its transcript is under Voice calls in the sidebar.",
                 icon=":material/call_end:")
