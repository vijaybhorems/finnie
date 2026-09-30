"""Visual layer: brand mark, global CSS, and the static sign-in page.

Colours and radius come from ``[theme]`` in ``.streamlit/config.toml``; this file
adds what the theme can't express (hiding Streamlit's chrome, card and chat
styling, the hero). Everything is inline — no web fonts or remote images — so
the first paint needs nothing beyond Streamlit's own bundle.
"""
from __future__ import annotations

import html

import streamlit as st

ACCENT = "#0E7C66"

# Rising-bars mark, drawn inline so the sidebar never waits on an image host.
LOGO_SVG = """
<svg width="{size}" height="{size}" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect width="32" height="32" rx="9" fill="url(#fg)"/>
  <rect x="8" y="17" width="4" height="7" rx="1.5" fill="#fff" opacity=".65"/>
  <rect x="14" y="13" width="4" height="11" rx="1.5" fill="#fff" opacity=".82"/>
  <rect x="20" y="8" width="4" height="16" rx="1.5" fill="#fff"/>
  <defs><linearGradient id="fg" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
    <stop stop-color="#34D399"/><stop offset="1" stop-color="#0E7C66"/></linearGradient></defs>
</svg>
"""

_CSS = """
<style>
/* ── Chrome ─────────────────────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 3rem; padding-bottom: 6rem; max-width: 1080px; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { border-right: none; }
[data-testid="stSidebar"] hr { margin: .9rem 0; opacity: .25; }
.fn-brand { display: flex; align-items: center; gap: .65rem; margin: .25rem 0 1.1rem; }
.fn-brand-name { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
.fn-brand-tag { font-size: .72rem; opacity: .6; margin-top: .25rem; letter-spacing: .02em; }

/* Navigation: the radio rendered as a vertical list of pill links */
[data-testid="stSidebar"] [role="radiogroup"] { gap: .2rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100%; padding: .55rem .8rem; border-radius: .6rem; cursor: pointer;
  transition: background .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(52,211,153,.14); font-weight: 600;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }

.fn-user { display: flex; align-items: center; gap: .6rem; }
.fn-user img, .fn-user .fn-avatar {
  width: 34px; height: 34px; border-radius: 50%; object-fit: cover; flex-shrink: 0;
}
.fn-user .fn-avatar {
  display: grid; place-items: center; background: #1E4A5C; font-weight: 600; font-size: .9rem;
}
.fn-user-name { font-weight: 600; font-size: .9rem; line-height: 1.2; }
.fn-user-mail { font-size: .75rem; opacity: .6; overflow: hidden; text-overflow: ellipsis; }
.fn-fineprint { font-size: .72rem; opacity: .55; line-height: 1.45; }

/* ── Page header ────────────────────────────────────────────────────────── */
.fn-hello { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; margin: 0; }
.fn-hello span {
  background: linear-gradient(90deg, #0E7C66, #34D399);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.fn-hello { line-height: 1.15; margin-top: 1.5rem; }
.fn-head { display: flex; align-items: center; gap: .9rem; margin: .5rem 0 1.4rem; }
.fn-head-icon {
  width: 2.9rem; height: 2.9rem; border-radius: .85rem; flex-shrink: 0;
  display: grid; place-items: center; font-size: 1.35rem;
  background: linear-gradient(145deg, #E3F6EE, #CDEFE2);
}
.fn-head-title { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.025em; line-height: 1.15; }
.fn-head-sub { color: #5B6B7A; font-size: .95rem; margin-top: .2rem; }
.fn-sub { color: #5B6B7A; margin: .5rem 0 1.75rem; font-size: 1.05rem; }

/* ── Suggestion cards (quick prompts) ───────────────────────────────────── */
.st-key-fn-prompts button {
  height: 100%; min-height: 4.4rem; justify-content: flex-start; text-align: left;
  padding: .8rem 1rem; border-radius: .9rem; background: #fff;
  box-shadow: 0 1px 2px rgba(15,27,45,.04);
  transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
}
.st-key-fn-prompts button:hover {
  border-color: #0E7C66; box-shadow: 0 6px 18px rgba(14,124,102,.12); transform: translateY(-1px);
}
.st-key-fn-prompts button p { font-size: .9rem; line-height: 1.35; }

/* ── Chat ───────────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] { padding: .9rem 1.1rem; border-radius: 1rem; gap: .9rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
  background: #F1F6F4;
}
.fn-chip {
  display: inline-flex; align-items: center; gap: .35rem; margin-top: .35rem;
  font-size: .74rem; font-weight: 500; color: #3E5566;
  background: #EEF2F5; border-radius: 999px; padding: .18rem .6rem;
}
.fn-chip.fast { background: #E3F6EE; color: #0B6B58; }
.fn-chip-why { font-weight: 400; opacity: .8; }
[data-testid="stChatInput"] { border-radius: 1rem; box-shadow: 0 8px 28px rgba(15,27,45,.08); }

/* ── Cards (metrics, containers with border) ────────────────────────────── */
[data-testid="stMetric"] {
  background: #fff; border: 1px solid #E4E8EC; border-radius: .9rem; padding: .75rem .85rem;
}
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; letter-spacing: -0.02em; }
[data-testid="stMetricLabel"] p { font-size: .8rem; color: #5B6B7A; }
/* Section headings sit below the page header, so keep them a step smaller */
.block-container h3 { font-size: 1.2rem; font-weight: 650; padding-top: .6rem; }
[data-testid="stExpander"] details { border-radius: .8rem; }
[data-testid="stPopover"] button p { white-space: nowrap; }
</style>
"""

_LOGIN_HTML = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { max-width: 1120px; padding-top: 7vh; }
@media (max-width: 860px) { .fn-preview { display: none; } }
.fn-eyebrow {
  display: inline-block; font-size: .75rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: #0E7C66; background: #E3F6EE; padding: .3rem .7rem; border-radius: 999px; margin: 1.4rem 0 1rem;
}
.fn-login h1 { font-size: 3rem; line-height: 1.05; letter-spacing: -0.035em; margin: 0 0 1rem; padding: 0; }
.fn-login h1 span {
  background: linear-gradient(90deg, #0E7C66, #34D399);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.fn-login .fn-lead { font-size: 1.1rem; color: #5B6B7A; line-height: 1.55; max-width: 34rem; }
.fn-feats { list-style: none; padding: 0; margin: 1.6rem 0 0; display: grid; gap: .7rem; }
.fn-feats li { display: flex; gap: .7rem; align-items: flex-start; color: #243447; margin: 0; }
.fn-feats b { display: block; }
.fn-feats small { color: #6B7B8A; font-size: .88rem; }
.fn-dot {
  width: 1.9rem; height: 1.9rem; border-radius: .6rem; background: #E3F6EE; flex-shrink: 0;
  display: grid; place-items: center; font-size: .95rem;
}
.fn-preview {
  background: linear-gradient(160deg, #0B1F2A, #12384A); border-radius: 1.4rem; padding: 1.4rem;
  box-shadow: 0 30px 60px rgba(11,31,42,.25); color: #E6EEF2;
}
.fn-preview .bar { display: flex; gap: .35rem; margin-bottom: 1.1rem; }
.fn-preview .bar i { width: .6rem; height: .6rem; border-radius: 50%; background: rgba(255,255,255,.18); }
.fn-bubble { padding: .75rem .95rem; border-radius: 1rem; font-size: .9rem; line-height: 1.5; margin-bottom: .75rem; }
.fn-bubble.me { background: rgba(52,211,153,.16); margin-left: 18%; border-bottom-right-radius: .3rem; }
.fn-bubble.ai { background: rgba(255,255,255,.07); margin-right: 8%; border-bottom-left-radius: .3rem; }
.fn-bubble.ai em { color: #6EE7B7; font-style: normal; font-weight: 600; }
.fn-tag { font-size: .7rem; opacity: .6; margin-top: .45rem; }
.fn-spark { display: flex; align-items: flex-end; gap: .3rem; height: 3.2rem; margin: .8rem 0 .2rem; }
.fn-spark span { flex: 1; border-radius: .25rem .25rem 0 0; background: linear-gradient(#34D399, rgba(52,211,153,.25)); }
.st-key-fn-signin button {
  background: #0B1F2A; color: #fff; border: none; border-radius: .8rem; padding: .7rem 1.2rem;
  font-weight: 600; box-shadow: 0 8px 20px rgba(11,31,42,.18);
}
.st-key-fn-signin button:hover { background: #12384A; color: #fff; }
.fn-legal { font-size: .78rem; color: #8A98A5; margin-top: .8rem; }
</style>
"""

_LOGIN_COPY = """
<div class="fn-login">
    {logo}
    <div class="fn-eyebrow">AI finance education</div>
    <h1>Money questions,<br><span>answered clearly.</span></h1>
    <p class="fn-lead">Finnie explains investing, markets, taxes and planning in plain language,
      grounded in a curated knowledge base and live market data.</p>
    <ul class="fn-feats">
      <li><div class="fn-dot">💬</div><div><b>Ask anything</b><small>Six specialist agents route every question to the right expert.</small></div></li>
      <li><div class="fn-dot">📊</div><div><b>See your portfolio</b><small>Allocation, concentration and performance at a glance.</small></div></li>
      <li><div class="fn-dot">🎯</div><div><b>Plan your goals</b><small>Project savings through life events like a home or a child.</small></div></li>
    </ul>
</div>
"""

_LOGIN_PREVIEW = """
  <div class="fn-preview">
    <div class="bar"><i></i><i></i><i></i></div>
    <div class="fn-bubble me">How does compound interest work?</div>
    <div class="fn-bubble ai">It's interest earned on your interest. $10,000 at 7% a year grows to
      about <em>$19,700</em> in 10 years and <em>$76,100</em> in 30 — most of it in the later years.
      <div class="fn-spark">
        <span style="height:12%"></span><span style="height:16%"></span><span style="height:21%"></span>
        <span style="height:27%"></span><span style="height:35%"></span><span style="height:45%"></span>
        <span style="height:58%"></span><span style="height:74%"></span><span style="height:100%"></span>
      </div>
      <div class="fn-tag">📚 Finance Q&amp;A</div>
    </div>
  </div>
"""


def _html(markup: str) -> str:
    """Flatten markup for st.markdown, which renders indented lines as code blocks."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def apply_theme() -> None:
    """Inject the global stylesheet. Call once per script run, after set_page_config."""
    st.markdown(_html(_CSS), unsafe_allow_html=True)


def logo(size: int = 36) -> str:
    return _html(LOGO_SVG.format(size=size))


def render_brand() -> None:
    st.markdown(
        f'<div class="fn-brand">{logo(38)}<div><div class="fn-brand-name">Finnie</div>'
        f'<div class="fn-brand-tag">AI finance education</div></div></div>',
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str = "", container=None) -> None:
    """The shared tab header: tinted icon tile, title, one-line subtitle."""
    sub = f'<div class="fn-head-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    (container or st).markdown(
        f'<div class="fn-head"><div class="fn-head-icon">{icon}</div><div>'
        f'<div class="fn-head-title">{html.escape(title)}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def render_login_page(on_sign_in) -> None:
    """The signed-out landing page: static copy, a preview card, one button."""
    st.markdown(_html(_LOGIN_HTML), unsafe_allow_html=True)
    copy_col, preview_col = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
    with copy_col:
        st.markdown(_html(_LOGIN_COPY.replace("{logo}", logo(44))), unsafe_allow_html=True)
        st.write("")
        with st.container(key="fn-signin"):
            if st.button("Continue with Google", icon=":material/login:"):
                on_sign_in()
        st.markdown(
            '<div class="fn-legal">Educational information only — not personalised financial advice.</div>',
            unsafe_allow_html=True,
        )
    with preview_col:
        st.markdown(_html(_LOGIN_PREVIEW), unsafe_allow_html=True)
