"""Plays a simulated-call scenario (tests/evals/sim_calls.py) against the real worker.

Needs the voice venv (livekit rtc), the local LiveKit server and macOS `say`:
    docker run -d --name finnie-livekit -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\
        livekit/livekit-server:v1.13.10 --dev --bind 0.0.0.0 --node-ip 127.0.0.1

`Worker` starts `src.voice.worker start` itself, with voice on for the test
email, an in-memory store (never the dev database), a local turn log, and any
voice settings to try (FINNIE_VOICE_SETTINGS). Stop any other worker first:
LiveKit would hand it some of the calls.

The caller's speech is macOS `say` (free), cached under .sim_cache/ by text,
voice, speed and noise. Each call is a new made-up user (g-local-sim…), so its
turns and summary are found by user id.
"""
from __future__ import annotations

import array
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Optional

from dotenv import dotenv_values

from tests.evals.sim_calls import FollowUp, Observed, Scenario, Silence, Speak

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).parent / ".sim_cache"
RATE = 16000
FRAME = RATE // 100  # 10 ms
QUIET_S = 0.3  # an answer must fall silent this long before new audio counts as a new reply
REPLY_START_S = 40.0  # longest wait for Finnie to start a reply
REPLY_END_S = 60.0  # longest wait for one reply to finish
OPENING_S = 150.0  # the first call waits for a process to load (~15-40 s)


def _env_file() -> dict[str, str]:
    return {k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None}


# ── The caller's voice ──────────────────────────────────────────────────────

def _say(text: str, voice: str, rate: int, out: Path) -> None:
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(out),
                    "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True)


def _pcm(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getsampwidth() == 2 and w.getnchannels() == 1, path
        return w.readframes(w.getnframes())


def synthesize(speak: Speak) -> bytes:
    """16 kHz mono PCM for `speak`: each text part spoken, each number a pause."""
    key = hashlib.sha256(json.dumps([speak.parts, speak.voice, speak.rate, speak.noise]).encode()).hexdigest()[:16]
    out = CACHE / f"{key}.wav"
    if out.exists():
        return _pcm(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    pcm = bytearray()
    with tempfile.TemporaryDirectory() as tmp:
        for i, part in enumerate(speak.parts):
            if isinstance(part, str):
                clip = Path(tmp) / f"{i}.wav"
                _say(part, speak.voice, speak.rate, clip)
                pcm += _pcm(clip)
            else:
                pcm += bytes(int(RATE * part) * 2)
        clean = Path(tmp) / "clean.wav"
        with wave.open(str(clean), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
            w.writeframes(bytes(pcm))
        if speak.noise:
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(clean),
                 "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude=0.03:sample_rate={RATE}",
                 "-filter_complex", "amix=inputs=2:duration=first:normalize=0", "-ar", str(RATE), str(out)],
                check=True,
            )
        else:
            clean.replace(out)
    return _pcm(out)


def follow_up_line(goal: str, transcript: list[tuple[str, str]]) -> str:
    """The LLM caller: one line a person might say next, from the call so far."""
    from langchain_anthropic import ChatAnthropic

    from src.core.config import get_settings

    model = ChatAnthropic(
        model=get_settings().voice.classifier_model,  # Haiku
        anthropic_api_key=_env_file().get("ANTHROPIC_API_KEY") or os.environ["ANTHROPIC_API_KEY"],
        max_tokens=80,
    )
    call = "\n".join(f"{who}: {text}" for who, text in transcript[-8:])
    reply = model.invoke([
        {"role": "system", "content": "You are a person on a phone call with Finnie, a finance-education "
                                      "assistant. Reply with only the words you say next, as plain speech."},
        {"role": "user", "content": f"The call so far:\n{call}\n\n{goal}"},
    ])
    return " ".join(str(reply.content).replace('"', "").split())


# ── The worker under test ───────────────────────────────────────────────────

class Worker:
    """`python -m src.voice.worker start` in a subprocess, with its log and turn log in a temp dir."""

    def __init__(self, settings: Optional[dict[str, Any]] = None, email: str = "tester@example.com") -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="finnie-sim-"))
        self.log = self.dir / "worker.log"
        self.turn_log = self.dir / "turns.jsonl"
        env = {**os.environ, **_env_file()}  # the real keys: tests mock the process's own env
        for key, dev in (("LIVEKIT_URL", "ws://localhost:7880"), ("LIVEKIT_API_KEY", "devkey"),
                         ("LIVEKIT_API_SECRET", "secret")):  # `livekit-server --dev`, as local_token.py
            env.setdefault(key, dev)
        env.update({
            "FINNIE_VOICE_ENABLED": "1",
            "FINNIE_VOICE_ALLOWED_EMAILS": email,
            "VOICE_TURN_LOG": str(self.turn_log),
            "FINNIE_VOICE_SETTINGS": json.dumps(settings or {}),
            "DATABASE_URL": "",  # in-memory: never the developer's database
            "PYTHONUNBUFFERED": "1",
        })
        self._fh = self.log.open("w")
        self.proc = subprocess.Popen([sys.executable, "-m", "src.voice.worker", "start"],
                                     cwd=ROOT, env=env, stdout=self._fh, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout: float = 120) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker exited: {self.log.read_text()[-2000:]}")
            if "registered worker" in self.log.read_text():
                return
            time.sleep(1)
        raise TimeoutError("worker never registered with LiveKit")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._fh.close()

    def _lines(self, path: Path) -> list[dict[str, Any]]:
        out = []
        for line in path.read_text().splitlines() if path.exists() else []:
            start = line.find("{")
            if start >= 0:
                try:
                    out.append(json.loads(line[start:]))
                except json.JSONDecodeError:
                    pass
        return out

    def turns(self, user_id: str) -> list[dict[str, Any]]:
        return [r for r in self._lines(self.turn_log) if r.get("user_id") == user_id]

    async def summary(self, user_id: str, timeout: float = 40) -> dict[str, Any]:
        """The call's voice_call_summary, logged once the worker has wrapped up (memory saves wait)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for r in self._lines(self.log):
                if r.get("event") == "voice_call_summary" and r.get("user_id") == user_id:
                    return r
            await asyncio.sleep(1)
        return {}


# ── The caller ──────────────────────────────────────────────────────────────

def _rms(data: bytes) -> float:
    samples = array.array("h", data)
    return math.sqrt(sum(s * s for s in samples) / max(1, len(samples)))


class Caller:
    def __init__(self, name: str) -> None:
        from livekit import rtc

        self.rtc = rtc
        self.name = name
        self.room = rtc.Room()
        self.state = "?"
        self.state_changed = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.observed = Observed()
        self._spoke_at: Optional[float] = None
        self._first_audio: Optional[float] = None
        self._quiet_since: Optional[float] = None

    async def connect(self) -> str:
        from spike.voice.local_token import LIVEKIT_URL, new_call
        from src.voice.page_events import TOPIC

        rtc = self.rtc
        identity, _room, token = new_call(self.name)
        self.identity = identity
        self.room.on("disconnected", lambda *_: self.disconnected.set())

        @self.room.on("participant_attributes_changed")
        def _attrs(changed: dict, participant: Any) -> None:
            if "lk.agent.state" in changed:
                self.state = changed["lk.agent.state"]
                self.state_changed.set()

        @self.room.on("track_subscribed")
        def _track(track: Any, *_: Any) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(self._listen(track))

        def _reader(kind: str):
            def handler(reader: Any, sender: str) -> None:
                async def read() -> None:
                    try:
                        text = (await reader.read_all()).strip()
                    except rtc.data_stream.StreamError:
                        return
                    if not text:
                        return
                    if kind == "transcript":
                        self.observed.transcript.append(("You" if sender == identity else "Finnie", text))
                    else:
                        event = json.loads(text)
                        if event.get("type") == "answer":
                            self.observed.answers.append(event)
                asyncio.ensure_future(read())
            return handler

        self.room.register_text_stream_handler("lk.transcription", _reader("transcript"))
        self.room.register_text_stream_handler(TOPIC, _reader("page"))
        await self.room.connect(LIVEKIT_URL, token)
        self.source = rtc.AudioSource(RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("mic", self.source)
        await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._spoken = asyncio.Event()
        asyncio.ensure_future(self._feed())
        return identity

    async def _listen(self, track: Any) -> None:
        async for event in self.rtc.AudioStream(track):
            if self._spoke_at is None or self._first_audio is not None:
                continue
            now = time.perf_counter()
            if _rms(bytes(event.frame.data)) <= 300:
                self._quiet_since = self._quiet_since or now
            elif self._quiet_since is not None and now - self._quiet_since >= QUIET_S:
                self._first_audio = now
            else:
                self._quiet_since = None

    async def _feed(self) -> None:
        """One task feeds the mic forever: queued speech, else silence (never cut mid-frame)."""
        rtc = self.rtc
        silence = bytes(FRAME * 2)
        while self.room.isconnected():
            if self._queue.empty():
                await self.source.capture_frame(rtc.AudioFrame(silence, RATE, 1, FRAME))
                continue
            pcm = self._queue.get_nowait()
            for i in range(0, len(pcm), FRAME * 2):
                chunk = pcm[i:i + FRAME * 2].ljust(FRAME * 2, b"\0")
                await self.source.capture_frame(rtc.AudioFrame(chunk, RATE, 1, FRAME))
            self._spoke_at, self._first_audio, self._quiet_since = time.perf_counter(), None, None
            self._spoken.set()

    async def _speak(self, pcm: bytes) -> None:
        self._spoken.clear()
        self._queue.put_nowait(pcm)
        await self._spoken.wait()

    async def wait_state(self, wanted: set[str], timeout: float) -> bool:
        deadline = time.perf_counter() + timeout
        while self.state not in wanted and not self.disconnected.is_set():
            self.state_changed.clear()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self.state_changed.wait(), remaining)
            except asyncio.TimeoutError:
                return False
        return self.state in wanted

    async def wait_reply_done(self) -> None:
        if self.state != "speaking":
            await self.wait_state({"thinking", "speaking"}, REPLY_START_S)
        while not self.disconnected.is_set():
            if not await self.wait_state({"listening"}, REPLY_END_S):
                return
            await asyncio.sleep(1.5)  # a short gap (preamble, then answer) isn't the end
            if self.state == "listening":
                return

    async def run(self, scenario: Scenario) -> None:
        await self.wait_state({"speaking"}, OPENING_S)  # the session-start disclaimer
        await self.wait_state({"listening"}, REPLY_END_S)
        for action in scenario.actions:
            if self.disconnected.is_set():
                return
            if isinstance(action, Silence):
                try:
                    await asyncio.wait_for(self.disconnected.wait(), action.seconds)
                    self.observed.ended_by_worker = True
                except asyncio.TimeoutError:
                    pass
                continue
            if isinstance(action, FollowUp):
                await asyncio.sleep(2)  # let the transcript stream catch up
                line = await asyncio.to_thread(follow_up_line, action.goal, self.observed.transcript)
                action = Speak((line,), voice=action.voice)
            self.observed.said.append(action.text)
            await self._speak(await asyncio.to_thread(synthesize, action))
            if action.wait == "answer":
                await self.wait_reply_done()
            elif action.wait == "speaking":
                await self.wait_state({"speaking"}, REPLY_START_S)
                await asyncio.sleep(action.then_s)
            if action.wait != "none":
                first = self._first_audio
                self.observed.first_audio_s.append(round(first - self._spoke_at, 2) if first else None)

    async def hang_up(self) -> None:
        if self.room.isconnected():
            await self.room.disconnect()


async def run_scenario(scenario: Scenario, worker: Worker, tag: str = "") -> Observed:
    caller = Caller(f"sim{scenario.name}{tag}")
    identity = await caller.connect()
    try:
        await caller.run(scenario)
        await asyncio.sleep(2)  # the last spoken item and page event arrive
    finally:
        await caller.hang_up()
    observed = caller.observed
    observed.summary = await worker.summary(identity)
    observed.turns = worker.turns(identity)
    return observed
