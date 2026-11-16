"""Voice on the Chat page (design doc, Page changes).

- src/voice/page_events.py: what the worker sends the caller's page.
- src/web_app/voice.py: Talk, the call view's server side, and why a call ended,
  run in Streamlit's AppTest with the token, the component and the user's
  settings stubbed. The browser half (voice_call/call.js) is checked by hand
  against a local LiveKit server; its markdown renderer is tested here in node.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src.voice.page_events import (
    SPEECH_DOWN, TOPIC, answer_event, encode, notice_event, spoken_event,
)
from src.voice.turns import SORRY

ROOM = "g-1:" + "a" * 32


class TestPageEvents:
    def test_answer_carries_the_full_written_answer(self):
        event = answer_event(2, "tax_education", "**Roth**: taxed now.\n\nDisclaimer: …")
        assert event == {"type": "answer", "turn": 2, "agent": "tax_education",
                         "answer": "**Roth**: taxed now.\n\nDisclaimer: …"}

    def test_a_failed_turn_shows_the_apology_not_an_error(self):
        assert answer_event(3, "error", "Traceback …")["answer"] == SORRY
        assert answer_event(3, None, None) == {"type": "answer", "turn": 3, "agent": "error", "answer": SORRY}
        assert answer_event(3, "finance_qa", "")["answer"] == SORRY

    def test_spoken_text_is_tidied_and_flags_an_interruption(self):
        assert spoken_event(1, "  The big  difference\nis tax. ", True) == {
            "type": "spoken", "turn": 1, "spoken": "The big difference is tax.", "interrupted": True}

    def test_notice_and_encoding(self):
        assert notice_event(SPEECH_DOWN) == {"type": "notice", "text": SPEECH_DOWN}
        assert json.loads(encode(answer_event(1, "finance_qa", "5 € — ok"))) ["answer"] == "5 € — ok"
        assert "€" in encode(notice_event("€"))  # readable, not \\u escapes
        assert TOPIC == "finnie.voice"  # call.js registers this topic


# ── The page's server side, in AppTest ────────────────────────────────────────

def _end_call_app(reason: str) -> None:
    import time

    import streamlit as st

    from src.web_app import voice

    st.session_state.setdefault("voice_call", {"room": "g-1:" + "a" * 32, "started": time.time() - 75})
    if st.button("end"):
        voice.end_call(reason)
    st.write("in call" if voice.in_call() else "typing")


@pytest.mark.parametrize("reason", ["mic_denied", "load_failed", "failed", "no_agent", "agent_left", "already_ended"])
def test_a_call_that_failed_says_why(reason):
    from src.web_app.voice import END_MESSAGES

    at = AppTest.from_function(_end_call_app, args=(reason,)).run()
    at.button[0].click().run()
    assert [t.value for t in at.toast] == [END_MESSAGES[reason]]
    assert at.markdown[-1].value == "typing"


def test_a_call_the_user_ended_gives_its_length():
    at = AppTest.from_function(_end_call_app, args=("user_ended",)).run()
    at.button[0].click().run()
    assert at.toast[0].value.startswith("Call ended after 1:1")  # 75 s, give or take a second
    assert "Voice calls" in at.toast[0].value


class FakeUserData:
    """The settings voice.py reads and writes; writes are recorded per test."""

    consented = False
    saved: list = []

    def voice_consented(self):
        return FakeUserData.consented

    def set_voice_consent(self):
        FakeUserData.consented = True

    def speak_amounts(self, thread_id=None):
        return False

    def set_speak_amounts(self, enabled, thread_id=None):
        FakeUserData.saved.append((enabled, thread_id))


@pytest.fixture
def page(monkeypatch):
    """Stub the signed-in user, their settings and the token server for voice.py."""
    from src.voice import tokens
    from src.voice.tokens import CallToken, VoiceRefused
    from src.web_app import voice

    state = {"minted": 0, "refuse": None, "mounted": None, "reply": {}}
    FakeUserData.consented, FakeUserData.saved = False, []

    def mint(user_id, email):
        if state["refuse"]:
            raise VoiceRefused(state["refuse"])
        state["minted"] += 1
        return CallToken(url="ws://lk", token="t", room=ROOM)

    def component():
        def mount(key, data, **callbacks):
            state["mounted"] = data
            return SimpleNamespace(**state["reply"])
        return mount

    monkeypatch.setattr(voice, "current_user_data", FakeUserData)
    monkeypatch.setattr(voice, "current_user_id", lambda: "g-1")
    monkeypatch.setattr(voice, "_email", lambda: "a@example.com")
    monkeypatch.setattr(voice, "_component", component)
    monkeypatch.setattr(tokens, "mint_call_token", mint)
    return state


def _talk_app() -> None:
    import streamlit as st

    from src.web_app import voice

    st.write("in call" if voice.in_call() else "typing")
    voice.render_talk_button(st)


def test_talk_shows_the_privacy_notice_before_the_first_call(page):
    at = AppTest.from_function(_talk_app).run()
    at.button(key="voice_talk").click().run()
    assert any("Where your voice goes" in m.value for m in at.markdown)
    assert page["minted"] == 0  # no call until "Start call"


def test_talk_starts_a_call_at_once_after_consent(page):
    FakeUserData.consented = True
    at = AppTest.from_function(_talk_app).run()
    at.button(key="voice_talk").click().run()
    assert page["minted"] == 1
    assert at.markdown[0].value == "in call"  # the same press shows the call view
    assert at.session_state.voice_call["room"] == ROOM


def test_a_refused_call_explains_and_keeps_the_chat(page):
    from src.voice.tokens import REFUSALS

    FakeUserData.consented = True
    page["refuse"] = "rate_limited"
    at = AppTest.from_function(_talk_app).run()
    at.button(key="voice_talk").click().run()
    assert [w.value for w in at.warning] == [REFUSALS["rate_limited"]]
    assert at.markdown[0].value == "typing"


def _session_app() -> None:
    import time

    import streamlit as st

    from src.web_app import voice

    if "seeded" not in st.session_state:  # a call in progress, once: ending it must stick
        st.session_state.seeded = True
        st.session_state.voice_call = {"url": "ws://lk", "token": "t", "room": "g-1:" + "a" * 32,
                                       "started": time.time()}
    if voice.in_call():
        voice.render_voice_session({"tax_education": "Tax Education"})
    st.write("in call" if voice.in_call() else "typing")


def test_the_call_view_gets_this_calls_settings(page):
    AppTest.from_function(_session_app).run()
    assert page["mounted"] == {"url": "ws://lk", "token": "t", "room": ROOM, "speak_amounts": False,
                               "labels": {"tax_education": "Tax Education"}}
    assert FakeUserData.saved == []


def test_the_call_bar_switch_saves_for_this_call_only(page):
    page["reply"] = {"speak_amounts": True}
    at = AppTest.from_function(_session_app).run()
    assert FakeUserData.saved == [(True, ROOM)]
    at.run()  # the next rerun reports the same value: saved once
    assert FakeUserData.saved == [(True, ROOM)]
    assert page["mounted"]["speak_amounts"] is True


def test_an_ended_call_brings_the_chat_back(page):
    page["reply"] = {"ended": "user_ended"}
    at = AppTest.from_function(_session_app).run()
    assert at.markdown[-1].value == "typing"
    assert "voice_call" not in at.session_state


# ── The markdown renderer (call.js), in node ─────────────────────────────────

CALL_JS = Path(__file__).resolve().parents[1] / "src" / "web_app" / "voice_call" / "call.js"


def _render(markdown: str) -> str:
    script = (f"import {{ renderMarkdown }} from {json.dumps(CALL_JS.as_uri())};"
              "let input = ''; process.stdin.on('data', (d) => input += d);"
              "process.stdin.on('end', () => process.stdout.write(renderMarkdown(input)));")
    done = subprocess.run(["node", "--input-type=module", "-e", script], input=markdown,
                          capture_output=True, text=True, timeout=30, check=True)
    return done.stdout


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
class TestRenderMarkdown:
    def test_the_answer_layout_survives(self):
        html = _render("### Roth vs traditional\n\n**Roth**: pay tax *now*.\n\n- one\n- two\n\n"
                       "| | Roth |\n|---|---|\n| Tax | now |\n\n---\n*Disclaimer: educational.*")
        assert '<p class="fv-h">Roth vs traditional</p>' in html
        assert "<strong>Roth</strong>: pay tax <em>now</em>." in html
        assert "<ul><li>one</li><li>two</li></ul>" in html
        assert "<th>Roth</th>" in html and "<td>now</td>" in html
        assert "<hr>" in html and "<em>Disclaimer: educational.</em>" in html

    def test_model_text_cannot_add_markup_or_links(self):
        html = _render('<img src=x onerror=alert(1)> [click](javascript:alert(1)) ![p](http://x/p.png) `<b>`')
        assert "<img" not in html and "<b>" not in html
        assert "&lt;img" in html and "javascript:" not in html and "http://x" not in html
        assert "click" in html and "<code>&lt;b&gt;</code>" in html

    def test_amounts_and_asterisks_in_prose(self):
        html = _render("Contribute $7,000 * 2 = $14,000 if both of you qualify.")
        assert html == "<p>Contribute $7,000 * 2 = $14,000 if both of you qualify.</p>"
