"""Voice spike worker: LiveKit Agents in front of Finnie's existing graph.

Each spoken turn goes Deepgram -> stream_workflow() -> ElevenLabs. The graph
runs in-process exactly as for the chat page: same classifier, guardrails,
agents and caches. Per-turn latency lands in spike/voice/runs/turns.jsonl.

Run (see spike/voice/README.md for keys and the local LiveKit server):
    .venv-voice/bin/python -m spike.voice.worker dev        # joins rooms
    .venv-voice/bin/python -m spike.voice.worker console    # laptop mic/speaker, no browser
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterable

from dotenv import load_dotenv

load_dotenv()
# Same defaults as the Talk page: `livekit-server --dev`. LiveKit Cloud values come from .env.
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")
# Voice can't afford the model thinking silently before it speaks (~9 s on goals
# questions). Applies to this process only: the classifier and every agent.
# Set FINNIE_LLM_THINKING=adaptive to measure the chat-page behaviour instead.
os.environ.setdefault("FINNIE_LLM_THINKING", "disabled")
# Same routing and scope accuracy as Sonnet 5 on the labelled eval cases, ~0.45 s
# faster per turn (spike/voice/compare_classifier.py).
os.environ.setdefault("FINNIE_CLASSIFIER_MODEL", "claude-haiku-5-5")

from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    cli,
    llm,
)
from livekit.agents.types import FlushSentinel  # noqa: E402
from livekit.plugins import deepgram, elevenlabs, silero  # noqa: E402
from livekit.plugins.turn_detector.english import EnglishModel  # noqa: E402

from spike.voice.prefetch import FILLERS, warm_shared_data  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("finnie.voice")
from spike.voice.speech import SESSION_DISCLAIMER, SpeechBuffer, strip_written_disclaimer  # noqa: E402

RUNS = Path(__file__).parent / "runs"
# Both on by default; set to 0 to measure without them.
SPECULATE = os.environ.get("VOICE_SPECULATE", "1") == "1"  # classify + warm data while the user speaks
FILLER = os.environ.get("VOICE_FILLER", "1") == "1"  # say a short line first on slow turns
WARM_EVERY_S = 60.0  # prices cache for 5 min; re-warm at most once a minute
TTS_MODEL = os.environ.get("VOICE_TTS_MODEL", "eleven_flash_v2_5")
# ElevenLabs' default voice unless one is chosen (see "Emotion control" in the design doc).
VOICE_ID = os.environ.get("VOICE_ELEVEN_VOICE_ID", "")

# Deepgram keyterm prompting for finance words it would otherwise mishear.
KEYTERMS = [
    "Roth", "Roth IRA", "IRA", "401(k)", "403(b)", "HSA", "ETF", "RSI", "MACD",
    "P/E ratio", "S&P 500", "Nasdaq", "Dow Jones", "dividend", "capital gains",
    "tax-loss harvesting", "wash sale", "FIRE", "Finnie",
]

_SENTINEL = object()


class GraphLLM(llm.LLM):
    """Placeholder so AgentSession agrees to generate replies.

    LiveKit refuses to reply ("trying to generate reply without an LLM model")
    when the session has no LLM, even with llm_node overridden. The override
    below does the real work, so chat() is never reached.
    """

    @property
    def model(self) -> str:
        return "finnie-graph"

    @property
    def provider(self) -> str:
        return "finnie"

    def chat(self, **_: Any) -> llm.LLMStream:
        raise RuntimeError("FinnieVoiceAgent.llm_node answers turns; chat() is unused")


class FinnieVoiceAgent(Agent):
    """Replaces the LLM step with a call into Finnie's graph (not LLMAdapter: see design doc)."""

    def __init__(self, user_id: str) -> None:
        super().__init__(instructions="Answers come from Finnie's workflow, not this prompt.")
        self.user_id = user_id
        self.thread_id = f"{user_id}:{uuid.uuid4()}"  # its own thread per session (Open question 1)
        self.turns: list[dict[str, Any]] = []
        # Speculation state for the turn being spoken.
        self._finals: list[str] = []
        self._interim = ""
        self._spec_running = False
        self._spec_pending: str | None = None
        self._spec_done: set[str] = set()
        self._last_warm = 0.0

    # ── While the user speaks ───────────────────────────────────────────────

    def on_user_speaking(self) -> None:
        """Warm the shared live data as soon as the user starts talking."""
        if not SPECULATE or time.monotonic() - self._last_warm < WARM_EVERY_S:
            return
        self._last_warm = time.monotonic()
        asyncio.ensure_future(self._warm())

    async def _warm(self) -> None:
        try:
            result = await asyncio.to_thread(warm_shared_data)
            logger.info("voice_data_warmed", **result)
        except Exception as exc:  # noqa: BLE001 — a failed warm only means a cold fetch later
            logger.warning("voice_data_warm_failed", error=str(exc))

    def on_transcript(self, text: str, is_final: bool) -> None:
        """Classify the transcript so far; the turn reuses it if the words match."""
        if not SPECULATE:
            return
        if is_final:
            self._finals.append(text.strip())
            self._interim = ""
        else:
            self._interim = text.strip()
        so_far = " ".join(t for t in [*self._finals, self._interim] if t)
        if len(so_far.split()) < 3:
            return
        if self._spec_running:
            self._spec_pending = so_far  # only the newest text matters
        else:
            asyncio.ensure_future(self._speculate(so_far))

    async def _speculate(self, text: str) -> None:
        from src.workflow.classify import _normalize, classify_query
        from src.workflow.guardrail import _matches_blocklist

        self._spec_running = True
        try:
            while text:
                key = _normalize(text)
                if key not in self._spec_done and not _matches_blocklist(text):
                    self._spec_done.add(key)
                    logger.info("voice_speculative_classify", words=len(key.split()))
                    try:
                        await asyncio.to_thread(classify_query, text)
                    except Exception:  # noqa: BLE001 — the turn classifies again and fails closed
                        pass
                text, self._spec_pending = self._spec_pending or "", None
        finally:
            self._spec_running = False

    # ── The turn ────────────────────────────────────────────────────────────

    async def _filler_for(self, question: str) -> str | None:
        """A short line to say first on slow turns, once the route is known."""
        from src.workflow.classify import classify_query
        from src.workflow.guardrail import _matches_blocklist

        if not FILLER or _matches_blocklist(question):
            return None
        try:
            # Usually instant: classified while the user was speaking. If not,
            # this is the turn's one classifier call; the graph reuses it.
            verdict = await asyncio.to_thread(classify_query, question)
        except Exception:  # noqa: BLE001 — no filler; the graph fails closed on its own call
            return None
        return FILLERS.get(verdict.agent) if verdict.on_topic else None

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str | FlushSentinel]:
        question = _last_user_text(chat_ctx)
        self._finals, self._interim = [], ""  # the next turn's transcript starts fresh
        if not question:
            return
        started = time.perf_counter()
        filler = await self._filler_for(question)
        verdict_s = round(time.perf_counter() - started, 3)
        if filler:
            yield filler
            # Speak it now. Without this, ElevenLabs' sentence splitter holds the
            # filler until the answer's first words arrive, which defeats it.
            yield FlushSentinel()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sink: dict[str, Any] = {}
        timing = {"started": time.perf_counter(), "first_token": None}

        def produce() -> None:
            from src.workflow.graph import stream_workflow

            try:
                for token in stream_workflow(
                    question, sink=sink, user_id=self.user_id, thread_id=self.thread_id,
                    channel="voice",
                ):
                    if timing["first_token"] is None:
                        timing["first_token"] = time.perf_counter()
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        # The graph is synchronous; run it beside the event loop. On interruption
        # LiveKit stops consuming this generator, and the run finishes on its own
        # (design doc: "When the user interrupts").
        threading.Thread(target=produce, name="finnie-turn", daemon=True).start()

        buffer = SpeechBuffer()
        completed = False
        try:
            while (token := await queue.get()) is not _SENTINEL:
                spoken = buffer.push(strip_written_disclaimer(token))
                if spoken.strip():
                    yield spoken
            tail = buffer.flush()
            if tail.strip():
                yield tail
            completed = True
        finally:
            # Runs on interruption too (LiveKit closes this generator), so an
            # interrupted turn logs its own timings, not the previous turn's.
            self.turns.append({
                "question": question,
                "agent": sink.get("agent_used"),
                "cache_hit": sink.get("cache_hit"),
                "finnie_first_token_s": _elapsed(timing["started"], timing["first_token"]),
                "finnie_total_s": round(time.perf_counter() - timing["started"], 3),
                "finnie_completed": completed,
                "filler": filler,
                "verdict_s": verdict_s,  # ~0 when classified while the user spoke
                "speculate": SPECULATE,
            })


def _last_user_text(chat_ctx: llm.ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if getattr(item, "role", None) == "user":
            return (item.text_content or "").strip()
    return ""


def _elapsed(start: float, end: float | None) -> float | None:
    return round(end - start, 3) if end is not None else None


def prewarm(proc: JobProcess) -> None:
    """Load models before taking calls, as warmup.py does for the web app."""
    proc.userdata["vad"] = silero.VAD.load()
    from src.web_app import warmup

    warmup.wait()


# Prewarm loads the embedding model and the graph (~20-40 s cold); LiveKit's
# default 10 s initialize timeout kills the process before it can take a call.
server = AgentServer(setup_fnc=prewarm, initialize_process_timeout=120)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    # Spike only: identity comes from the token the talk page minted. Production
    # re-checks the g-<sub> form, room prefix and allowlist (design doc).
    agent = FinnieVoiceAgent(user_id=f"spike-{participant.identity}")

    tts_kwargs: dict[str, Any] = {"model": TTS_MODEL}
    if VOICE_ID:
        tts_kwargs["voice_id"] = VOICE_ID
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en-US", keyterm=KEYTERMS, smart_format=True),
        llm=GraphLLM(),
        tts=elevenlabs.TTS(**tts_kwargs),
        vad=ctx.proc.userdata["vad"],
        turn_handling={
            "turn_detection": EnglishModel(),
            # On by default in livekit-agents 1.8; the design keeps it off because
            # a discarded run would still write the checkpoint, FAQ cache and memories.
            "preemptive_generation": {"enabled": False},
        },
    )

    RUNS.mkdir(exist_ok=True)
    log = RUNS / "turns.jsonl"
    pending_user: dict[str, Any] = {}

    @session.on("conversation_item_added")
    def _on_item(event: Any) -> None:
        item = event.item
        role = getattr(item, "role", None)  # handoff records have no role
        metrics = dict(getattr(item, "metrics", None) or {})
        if role == "user":
            pending_user.clear()
            pending_user.update(metrics)
            return
        if role != "assistant" or not agent.turns:
            return
        record = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tts_model": TTS_MODEL,
            **agent.turns[-1],
            "end_of_turn_delay_s": pending_user.get("end_of_turn_delay"),
            "transcription_delay_s": pending_user.get("transcription_delay"),
            "llm_node_ttft_s": metrics.get("llm_node_ttft"),
            "llm_node_ttfs_s": metrics.get("llm_node_ttfs"),
            "tts_ttfb_s": metrics.get("tts_node_ttfb"),
            "interrupted": bool(getattr(item, "interrupted", False)),
        }
        parts = [record[k] for k in ("end_of_turn_delay_s", "llm_node_ttfs_s", "tts_ttfb_s")]
        # End of speech -> first audio: turn decision + first sentence to TTS + TTS first byte.
        record["time_to_first_audio_s"] = (
            round(sum(parts), 3) if all(p is not None for p in parts) else None
        )
        with log.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    @session.on("user_input_transcribed")
    def _on_transcript(event: Any) -> None:
        agent.on_transcript(event.transcript, event.is_final)

    @session.on("user_state_changed")
    def _on_user_state(event: Any) -> None:
        if event.new_state == "speaking":
            agent.on_user_speaking()

    await session.start(agent=agent, room=ctx.room)
    await session.say(SESSION_DISCLAIMER, allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
