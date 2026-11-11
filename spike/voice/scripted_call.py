"""A scripted caller: speaks questions into a room and times Finnie's replies.

No microphone needed. Each question is synthesized with ElevenLabs, published
as the caller's microphone track, and followed by silence so the turn ends.
The caller measures what a user hears: end of their question -> first audible
frame of Finnie's reply. The second question interrupts the first answer, to
exercise the interruption path. Finnie's audio is saved to runs/ as WAV.

Needs the LiveKit dev server and `python -m spike.voice.worker dev` running.
    .venv-voice/bin/python -m spike.voice.scripted_call
Costs: a few ElevenLabs and Deepgram seconds, one Finnie turn per question.
"""
from __future__ import annotations

import array
import asyncio
import json
import math
import os
import time
import uuid
import wave
from datetime import timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
RATE = 16000
FRAME = RATE // 100  # 10 ms
RUNS = Path(__file__).parent / "runs"
CALLER_VOICE = "EXAVITQu4vr4xnSDxMaL"  # a premade voice, distinct from Finnie's

# (question, seconds to listen before the next question; None = until Finnie stops)
SCRIPT = [
    ("What is a P/E ratio?", 12),  # cut off: the next question interrupts it
    ("How is Apple stock doing today?", 12),  # interrupted too
    ("How much do I need to save to retire at 55?", 12),  # goals: where thinking cost ~9 s
    ("What is a P/E ratio?", None),  # repeat: should be an FAQ cache hit
]
ANSWER_CAP_S = 40  # stop listening to one answer after this long
QUIET_S = 0.3  # an interrupted answer must fall silent this long before a new reply counts


def synthesize(text: str) -> bytes:
    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{CALLER_VOICE}",
        params={"output_format": f"pcm_{RATE}"},
        headers={"xi-api-key": os.environ["ELEVEN_API_KEY"]},
        json={"text": text, "model_id": "eleven_flash_v2_5"},
        timeout=30,
    )
    r.raise_for_status()
    return r.content


def rms(frame: rtc.AudioFrame) -> float:
    samples = array.array("h", bytes(frame.data))
    return math.sqrt(sum(s * s for s in samples) / max(1, len(samples)))


class Call:
    def __init__(self) -> None:
        self.room = rtc.Room()
        self.state = "?"
        self.state_changed = asyncio.Event()
        self.listen_from: float | None = None  # end of our question
        self.first_audio_at: float | None = None
        self.quiet_since: float | None = None
        self.heard = bytearray()
        self.heard_rate = 48000
        self.transcript: list[tuple[str, str]] = []

    async def connect(self) -> None:
        room_name = f"spike-scripted:{uuid.uuid4()}"
        token = (
            api.AccessToken(KEY, SECRET)
            .with_identity("scripted")
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .with_ttl(timedelta(minutes=10))
            .to_jwt()
        )

        @self.room.on("participant_attributes_changed")
        def _attrs(changed: dict, participant: rtc.Participant) -> None:
            if "lk.agent.state" in changed:
                self.state = changed["lk.agent.state"]
                self.state_changed.set()

        @self.room.on("track_subscribed")
        def _track(track: rtc.Track, *_: object) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(self._listen(track))

        def _text(reader: rtc.TextStreamReader, identity: str) -> None:
            async def read() -> None:
                try:
                    text = (await reader.read_all()).strip()
                except rtc.data_stream.StreamError:  # we hung up mid-sentence
                    return
                if text:
                    self.transcript.append(("You" if identity == "scripted" else "Finnie", text))
            asyncio.ensure_future(read())

        self.room.register_text_stream_handler("lk.transcription", _text)
        await self.room.connect(URL, token)
        self.source = rtc.AudioSource(RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("mic", self.source)
        await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

    async def _listen(self, track: rtc.Track) -> None:
        async for event in rtc.AudioStream(track):
            frame = event.frame
            self.heard_rate = frame.sample_rate
            self.heard.extend(bytes(frame.data))
            if not self.listen_from or self.first_audio_at is not None:
                continue
            now = time.perf_counter()
            if rms(frame) <= 300:
                self.quiet_since = self.quiet_since or now
            elif self.quiet_since is not None and now - self.quiet_since >= QUIET_S:
                # Loud after a real gap: the new reply, not the tail of an interrupted one.
                self.first_audio_at = now
            else:
                self.quiet_since = None

    async def wait_state(self, wanted: set[str], timeout: float) -> bool:
        deadline = time.perf_counter() + timeout
        while self.state not in wanted:
            self.state_changed.clear()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self.state_changed.wait(), remaining)
            except asyncio.TimeoutError:
                return False
        return True

    def start_mic(self) -> None:
        """One task feeds the track forever: queued speech, else silence.

        Never cancelled mid-frame (that leaves the source in an invalid state);
        the worker's VAD and turn detector decide when a question has ended.
        """
        self.pending: asyncio.Queue[bytes] = asyncio.Queue()
        self.spoken = asyncio.Event()
        asyncio.ensure_future(self._feed())

    async def _feed(self) -> None:
        silence = bytes(FRAME * 2)
        while self.room.isconnected():
            if self.pending.empty():
                await self.source.capture_frame(rtc.AudioFrame(silence, RATE, 1, FRAME))
                continue
            pcm = self.pending.get_nowait()
            for i in range(0, len(pcm), FRAME * 2):
                chunk = pcm[i : i + FRAME * 2].ljust(FRAME * 2, b"\0")
                await self.source.capture_frame(rtc.AudioFrame(chunk, RATE, 1, FRAME))
            self.listen_from = time.perf_counter()
            self.first_audio_at = None
            self.quiet_since = None
            self.spoken.set()

    async def say(self, pcm: bytes) -> None:
        self.spoken.clear()
        self.pending.put_nowait(pcm)
        await self.spoken.wait()


async def main() -> None:
    RUNS.mkdir(exist_ok=True)
    audio = [(q, synthesize(q)) for q, _ in SCRIPT]
    call = Call()
    await call.connect()
    call.start_mic()
    print("connected; waiting for Finnie's opening line")
    await call.wait_state({"speaking"}, 150)  # first job prewarms
    await call.wait_state({"listening"}, 60)

    results = []
    for (question, listen_s), (_, pcm) in zip(SCRIPT, audio):
        print(f"\nasking: {question}")
        await call.say(pcm)
        start = call.listen_from
        got_audio = False
        deadline = start + 60
        while time.perf_counter() < deadline and call.first_audio_at is None:
            await asyncio.sleep(0.05)
        got_audio = call.first_audio_at is not None
        first = round(call.first_audio_at - start, 2) if got_audio else None
        print(f"  first audible reply after {first}s (from end of question)")
        if listen_s is not None:
            await asyncio.sleep(listen_s)
        else:
            await asyncio.sleep(2)
            await call.wait_state({"listening"}, ANSWER_CAP_S)
        results.append({"question": question, "caller_time_to_first_audio_s": first,
                        "interrupted_next": listen_s is not None})

    await asyncio.sleep(2)
    await call.room.disconnect()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    wav_path = RUNS / f"scripted-{stamp}.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(call.heard_rate)
        w.writeframes(bytes(call.heard))
    with (RUNS / "scripted_calls.jsonl").open("a") as fh:
        for r in results:
            fh.write(json.dumps({"at": stamp, **r}) + "\n")

    print("\ntranscript:")
    for who, text in call.transcript:
        print(f"  {who}: {text[:300]}")
    print(f"\nFinnie's audio: {wav_path}")


if __name__ == "__main__":
    asyncio.run(main())
