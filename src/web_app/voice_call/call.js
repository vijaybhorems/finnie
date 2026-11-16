// Finnie voice call: the call bar and the live session view (design doc, Page changes).
//
// Runs in the page as an st.components.v2 module. The LiveKit room lives at
// module level, so a Streamlit rerun (the "Speak amounts" switch causes one)
// calls this function again and it re-binds to the running call instead of
// reconnecting. The call ends on End, on unmount (leaving the Chat page), or
// when the worker hangs up; Python hears about it once, through `ended`.
//
// What it shows comes from LiveKit only, never a Streamlit rerun per line:
// the user's live transcript (lk.transcription), the agent's state
// (lk.agent.state), and the worker's messages on "finnie.voice": the full
// written answer, the line actually spoken, and notices.

const LIVEKIT_URL = "https://cdn.jsdelivr.net/npm/livekit-client@2.23.0/dist/livekit-client.esm.mjs";
const ENDED_KEY = "finnie-voice-ended";
const AGENT_JOIN_MS = 45000; // a cold worker takes ~15 s to start a call
const AGENT_LEFT_GRACE_MS = 3000; // a room being deleted disconnects us within this
const STATUS = {
  connecting: "Connecting…",
  initializing: "Connecting…",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  paused: "Call paused",
  ended: "Call ended",
};

let call = null; // the one running call in this browser tab

export default async function (component) {
  const { data, parentElement, setStateValue, setTriggerValue } = component;
  const ui = bindUi(parentElement, data);

  if (wasEnded(data.room)) {
    // A call that already ended here (the page was left and came back): don't rejoin.
    setTriggerValue("ended", "already_ended");
    return () => {};
  }
  if (call && call.roomName === data.room) {
    call.attach(ui, setStateValue, setTriggerValue);
    return () => call && call.end("left_page");
  }
  if (call) await call.end("replaced");

  call = new Call(data, ui, setStateValue, setTriggerValue);
  call.start();
  return () => call && call.end("left_page");
}

function wasEnded(room) {
  try {
    return (sessionStorage.getItem(ENDED_KEY) || "").split(" ").includes(room);
  } catch {
    return false;
  }
}

function markEnded(room) {
  try {
    const ended = (sessionStorage.getItem(ENDED_KEY) || "").split(" ").filter(Boolean).slice(-20);
    sessionStorage.setItem(ENDED_KEY, [...ended, room].join(" "));
  } catch {
    /* private mode: the worker still refuses a used token's room */
  }
}

function bindUi(root, data) {
  const $ = (sel) => root.querySelector(sel);
  const ui = {
    log: $(".fv-log"),
    status: $(".fv-status-text"),
    dot: $(".fv-dot"),
    timer: $(".fv-timer"),
    meter: $(".fv-meter-fill"),
    notice: $(".fv-notice"),
    speak: $(".fv-speak input"),
    mute: $(".fv-mute"),
    end: $(".fv-end"),
    unlock: $(".fv-unlock"),
    labels: data.labels || {},
  };
  if (!ui.speak.dataset.bound) ui.speak.checked = Boolean(data.speak_amounts);
  return ui;
}

class Call {
  constructor(data, ui, setStateValue, setTriggerValue) {
    this.data = data;
    this.roomName = data.room;
    this.room = null;
    this.ended = false;
    this.muted = false;
    this.startedAt = null;
    this.cards = new Map(); // turn -> card element
    this.segments = new Map(); // transcript segment id -> line element
    this.audio = [];
    this.attach(ui, setStateValue, setTriggerValue);
  }

  attach(ui, setStateValue, setTriggerValue) {
    this.ui = ui;
    this.setStateValue = setStateValue;
    this.setTriggerValue = setTriggerValue;
    ui.end.onclick = () => this.end("user_ended");
    ui.mute.onclick = () => this.toggleMute();
    ui.unlock.onclick = () => this.room && this.room.startAudio().then(() => (ui.unlock.hidden = true));
    ui.speak.onchange = () => this.setStateValue("speak_amounts", ui.speak.checked);
    ui.speak.dataset.bound = "1";
    this.setStatus(this.status || "connecting");
  }

  async start() {
    let lk;
    try {
      lk = await import(LIVEKIT_URL);
    } catch (err) {
      this.fail("load_failed");
      return;
    }
    this.lk = lk;
    const room = new lk.Room({
      audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.room = room;
    room
      .on(lk.RoomEvent.TrackSubscribed, (track) => this.onTrack(track))
      .on(lk.RoomEvent.ParticipantAttributesChanged, (_changed, participant) => this.onAttributes(participant))
      .on(lk.RoomEvent.AudioPlaybackStatusChanged, () => (this.ui.unlock.hidden = room.canPlaybackAudio))
      .on(lk.RoomEvent.ParticipantConnected, (participant) => this.onJoined(participant))
      .on(lk.RoomEvent.ParticipantDisconnected, (participant) => this.onLeft(participant))
      .on(lk.RoomEvent.Disconnected, (reason) => this.end(reasonName(lk, reason)));
    room.registerTextStreamHandler("lk.transcription", (reader, from) => this.onTranscript(reader, from));
    room.registerTextStreamHandler("finnie.voice", (reader) => this.onEvent(reader));

    try {
      await room.connect(this.data.url, this.data.token);
      await room.localParticipant.setMicrophoneEnabled(true);
    } catch (err) {
      const denied = err && (err.name === "NotAllowedError" || /permission/i.test(String(err.message)));
      this.fail(denied ? "mic_denied" : "failed");
      return;
    }
    this.startedAt = Date.now();
    this.ui.unlock.hidden = room.canPlaybackAudio;
    this.timer = setInterval(() => this.tick(), 250);
    this.startMeter();
    this.onVisibility = () => this.setStatus(document.hidden ? "paused" : this.agentState || "listening");
    document.addEventListener("visibilitychange", this.onVisibility);
    for (const p of room.remoteParticipants.values()) this.onJoined(p);
    // Finnie joins a few seconds after the page. If it never does, say so rather than
    // leave the user talking to an empty room.
    if (!this.agentHere) this.joinTimer = setTimeout(() => this.agentHere || this.fail("no_agent"), AGENT_JOIN_MS);
  }

  isAgent(participant) {
    return Boolean(participant) && (participant.isAgent || participant.kind === this.lk.ParticipantKind?.AGENT);
  }

  onJoined(participant) {
    if (!this.isAgent(participant)) return;
    this.agentHere = true;
    clearTimeout(this.joinTimer);
    this.onAttributes(participant);
  }

  onLeft(participant) {
    // The worker ends a call by deleting the room, which disconnects us anyway; Finnie
    // leaving on its own means the worker stopped, and nobody is listening any more.
    if (!this.isAgent(participant) || this.ended) return;
    const anotherAgent = [...this.room.remoteParticipants.values()].some((p) => this.isAgent(p));
    if (!anotherAgent) setTimeout(() => this.ended || this.fail("agent_left"), AGENT_LEFT_GRACE_MS);
  }

  onTrack(track) {
    if (track.kind !== "audio") return;
    const el = track.attach();
    el.style.display = "none";
    document.body.appendChild(el);
    this.audio.push({ track, el });
  }

  onAttributes(participant) {
    const state = participant && participant.attributes && participant.attributes["lk.agent.state"];
    if (!state) return;
    this.agentState = state;
    if (!document.hidden) this.setStatus(state);
  }

  async onTranscript(reader, from) {
    // Only the user's words: Finnie's answer arrives whole, as a card.
    if (!this.room || from.identity !== this.room.localParticipant.identity) return;
    const attrs = reader.info.attributes || {};
    const id = attrs["lk.segment_id"] || reader.info.id;
    const text = (await reader.readAll()).trim();
    if (!text) return;
    let line = this.segments.get(id);
    if (!line) {
      line = el("div", "fv-user");
      this.segments.set(id, line);
      this.append(line);
    }
    line.textContent = text;
    line.classList.toggle("fv-interim", attrs["lk.transcription_final"] !== "true");
  }

  async onEvent(reader) {
    let event;
    try {
      event = JSON.parse(await reader.readAll());
    } catch {
      return;
    }
    if (event.type === "answer") this.showAnswer(event);
    else if (event.type === "spoken") this.showSpoken(event);
    else if (event.type === "notice") this.showNotice(event.text);
  }

  card(turn) {
    let card = this.cards.get(turn);
    if (!card) {
      card = el("div", turn === 0 ? "fv-card fv-opening" : "fv-card");
      card.append(el("div", "fv-answer"), el("div", "fv-meta"), el("div", "fv-said"));
      this.cards.set(turn, card);
      this.append(card);
    }
    return card;
  }

  showAnswer({ turn, agent, answer }) {
    const card = this.card(turn);
    card.querySelector(".fv-answer").innerHTML = renderMarkdown(answer || "");
    const label = this.ui.labels[agent];
    card.querySelector(".fv-meta").innerHTML = label ? `<span class="fv-chip">${escapeHtml(label)}</span>` : "";
  }

  showSpoken({ turn, spoken, interrupted }) {
    if (!spoken && !interrupted) return;
    const card = this.card(turn);
    if (turn === 0) {
      card.querySelector(".fv-answer").textContent = spoken;
      return;
    }
    const said = card.querySelector(".fv-said");
    said.innerHTML = `<span class="fv-said-label">Said aloud</span> ${escapeHtml(spoken || "")}`
      + (interrupted ? ' <span class="fv-chip fv-interrupted">interrupted</span>' : "");
  }

  showNotice(text) {
    this.ui.notice.textContent = text;
    this.ui.notice.hidden = !text;
  }

  append(node) {
    // The page scrolls, not the log: bring each new line into view above the call bar
    // (scroll-margin in call.css), as a chat does.
    this.ui.log.appendChild(node);
    node.scrollIntoView({ block: "end", behavior: "smooth" });
  }

  setStatus(status) {
    this.status = status;
    this.ui.status.textContent = STATUS[status] || status;
    this.ui.dot.dataset.state = status;
  }

  tick() {
    if (!this.startedAt) return;
    const s = Math.floor((Date.now() - this.startedAt) / 1000);
    this.ui.timer.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  startMeter() {
    const pub = this.room.localParticipant.getTrackPublication(this.lk.Track.Source.Microphone);
    const media = pub && pub.track && pub.track.mediaStreamTrack;
    if (!media || !window.AudioContext) return;
    this.audioCtx = new AudioContext();
    const analyser = this.audioCtx.createAnalyser();
    analyser.fftSize = 512;
    this.audioCtx.createMediaStreamSource(new MediaStream([media])).connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);
    const draw = () => {
      if (this.ended) return;
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += (v - 128) * (v - 128);
      const level = this.muted ? 0 : Math.min(1, Math.sqrt(sum / buf.length) / 40);
      this.ui.meter.style.width = `${Math.round(level * 100)}%`;
      this.meterFrame = requestAnimationFrame(draw);
    };
    draw();
  }

  async toggleMute() {
    if (!this.room) return;
    this.muted = !this.muted;
    await this.room.localParticipant.setMicrophoneEnabled(!this.muted);
    this.ui.mute.textContent = this.muted ? "Unmute" : "Mute";
    this.ui.mute.setAttribute("aria-pressed", String(this.muted));
  }

  // The page goes back to the chat at once, so the reason travels with `ended`
  // and the server says it in a toast (voice.py, END_MESSAGES).
  fail(reason) {
    this.end(reason);
  }

  async end(reason) {
    if (this.ended) return;
    this.ended = true;
    markEnded(this.roomName);
    clearInterval(this.timer);
    clearTimeout(this.joinTimer);
    cancelAnimationFrame(this.meterFrame);
    if (this.onVisibility) document.removeEventListener("visibilitychange", this.onVisibility);
    if (this.audioCtx) this.audioCtx.close().catch(() => {});
    for (const { track, el: audioEl } of this.audio) {
      track.detach(audioEl);
      audioEl.remove();
    }
    this.setStatus("ended");
    if (this.room) await this.room.disconnect().catch(() => {});
    if (call === this) call = null;
    if (reason !== "left_page") this.setTriggerValue("ended", reason);
  }
}

function reasonName(lk, reason) {
  const names = lk.DisconnectReason || {};
  for (const [name, value] of Object.entries(names)) if (value === reason) return String(name).toLowerCase();
  return "disconnected";
}

function el(tag, cls) {
  const node = document.createElement(tag);
  node.className = cls;
  return node;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// The written answer, as the chat page shows it: headings, emphasis, lists,
// tables and rules. Everything is escaped first; links keep only their text.
function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/(\*\*|__)(.+?)\1/g, "<strong>$2</strong>")
    .replace(/(^|[^\w*])\*(?!\s)(.+?)(?<!\s)\*/g, "$1<em>$2</em>");
}

export function renderMarkdown(markdown) {
  const out = [];
  const lines = markdown.replace(/\r/g, "").split("\n");
  let list = null;
  let para = [];
  const flushPara = () => {
    if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`);
    list = null;
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      flushPara();
      flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = cells(line);
      const rows = [];
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      i--;
      out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`
        + rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody></table>");
    } else if (/^\s*-{3,}\s*$/.test(line)) {
      flushPara();
      flushList();
      out.push("<hr>");
    } else if (heading) {
      flushPara();
      flushList();
      out.push(`<p class="fv-h">${inline(heading[2])}</p>`);
    } else if (bullet || numbered) {
      flushPara();
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tag !== tag) {
        flushList();
        list = { tag, items: [] };
      }
      list.items.push((bullet || numbered)[1]);
    } else if (!line.trim()) {
      flushPara();
      flushList();
    } else {
      flushList();
      para.push(line.replace(/^\s*>\s?/, ""));
    }
  }
  flushPara();
  flushList();
  return out.join("");
}
