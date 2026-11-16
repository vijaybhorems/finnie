"""Voice worker: LiveKit Agents in front of Finnie's existing graph.

Each spoken turn goes Deepgram -> stream_workflow() -> ElevenLabs. The graph
runs in-process exactly as for the chat page: same classifier, guardrails,
agents and caches, with channel="voice" for short spoken answers.

A call joins only if src/voice/access.py allows it: voice enabled, a g-<sub>
identity, a room named after that identity, and an allowlisted email. The room
name is the call's conversation thread. A refused call hears VOICE_UNAVAILABLE,
ends, and leaves nothing saved.

Needs requirements-voice.txt on top of requirements.txt (spike/voice/README.md):
    .venv-voice/bin/python -m src.voice.worker dev        # joins rooms
    .venv-voice/bin/python -m src.voice.worker console    # laptop mic/speaker, no browser
Set VOICE_TURN_LOG=spike/voice/runs/turns.jsonl to keep per-turn timings with the question text.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterable

from dotenv import load_dotenv

load_dotenv()
# Same defaults as `livekit-server --dev`. LiveKit Cloud values come from .env.
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")

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

from src.core.config import get_settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.voice.access import VOICE_UNAVAILABLE, check_join, new_voice_room  # noqa: E402
from src.voice.limits import DAILY_LIMIT, ENDS_CALL, LINES, TURN_LIMIT, CallMonitor  # noqa: E402
from src.voice.prefetch import FILLERS, warm_shared_data  # noqa: E402
from src.voice.speech import SESSION_DISCLAIMER, SpeechBuffer, strip_written_disclaimer  # noqa: E402

logger = get_logger("finnie.voice")

# Both on by default; set to 0 to measure without them.
SPECULATE = os.environ.get("VOICE_SPECULATE", "1") == "1"  # classify + warm data while the user speaks
FILLER = os.environ.get("VOICE_FILLER", "1") == "1"  # say a short line first on slow turns
WARM_EVERY_S = 60.0  # prices cache for 5 min; re-warm at most once a minute
USAGE_FLUSH_S = 30.0  # save voice seconds this often, so a crashed call still counts
TURN_LOG = os.environ.get("VOICE_TURN_LOG", "")  # local measurement only: holds what users said


def configure_process() -> None:
    """Apply the voice settings that differ from chat to this process only.

    Must run before the first get_llm() or get_pool() call, which cache what
    they read. FINNIE_LLM_THINKING / FINNIE_CLASSIFIER_MODEL still win when set,
    so the chat-page behaviour can be measured from the worker.
    """
    settings = get_settings()
    voice = settings.voice
    # Voice can't afford the model thinking silently before it speaks (~9 s on goals questions).
    if not os.environ.get("FINNIE_LLM_THINKING"):
        settings.llm.thinking = voice.llm_thinking
    # Same routing and scope accuracy as Sonnet 5 on the labelled cases, ~0.45 s
    # faster per turn (spike/voice/compare_classifier.py).
    if not os.environ.get("FINNIE_CLASSIFIER_MODEL"):
        settings.llm.classifier_model = voice.classifier_model
    settings.persistence.pool_max_size = min(settings.persistence.pool_max_size, voice.db_pool_max_size)


configure_process()

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

    def __init__(self, user_id: str, thread_id: str) -> None:
        super().__init__(instructions="Answers come from Finnie's workflow, not this prompt.")
        self.user_id = user_id
        self.thread_id = thread_id  # the room name: its own thread per call (Open question 1)
        self.monitor: CallMonitor | None = None  # set by the entrypoint; None in tests
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
        if self.monitor is not None:
            if self.monitor.turn_limit_reached:
                return  # the goodbye is already said; the call is ending
            if not self.monitor.start_turn():
                yield TURN_LIMIT  # instead of answering; _enforce_limits ends the call after it
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


async def _refuse(ctx: JobContext, reason: str, line: str = VOICE_UNAVAILABLE) -> None:
    """Say `line` and end the call. Nothing is saved for a refused call."""
    logger.warning("voice_join_refused", reason=reason)
    session = AgentSession(tts=elevenlabs.TTS(model=get_settings().voice.tts_model))
    try:
        await session.start(agent=Agent(instructions="Refusing this call."), room=ctx.room)
        await session.say(line, allow_interruptions=False, add_to_chat_ctx=False)
    except Exception as exc:  # noqa: BLE001 — the call still ends, just without the line
        logger.warning("voice_refusal_line_failed", error=type(exc).__name__)
    finally:
        await session.aclose()
        await ctx.delete_room()
        ctx.shutdown(reason=f"refused: {reason}")


class _Usage:
    """Voice seconds not yet saved to the user's daily total."""

    def __init__(self, data: Any, thread_id: str, started: float) -> None:
        self._data, self._thread_id, self.started = data, thread_id, started
        self.saved_until = started
        self._finished = False

    async def save(self, now: float) -> None:
        delta, self.saved_until = now - self.saved_until, now  # claimed before awaiting
        if delta > 0:
            await asyncio.to_thread(self._data.add_voice_seconds, delta)

    async def finish(self, now: float) -> None:
        """Save the rest and mark the call ended; once, however the call ends."""
        if self._finished:
            return
        self._finished = True
        try:
            await self.save(now)
            await asyncio.to_thread(self._data.end_voice_call, self._thread_id, now - self.started)
        except Exception as exc:  # noqa: BLE001 — never block shutdown on bookkeeping
            logger.warning("voice_usage_save_failed", error=type(exc).__name__)


async def _end_call(session: AgentSession, reason: str) -> None:
    """End the call from the worker's side; the session's close handler wraps up."""
    logger.info("voice_call_ending", reason=reason)
    await session.aclose()


async def _enforce_limits(session: AgentSession, monitor: CallMonitor, usage: _Usage) -> None:
    """Once a second: save usage now and then, and act on the call's limits."""
    while True:
        await asyncio.sleep(1)
        now = time.monotonic()
        if now - usage.saved_until >= USAGE_FLUSH_S:
            try:
                await usage.save(now)
            except Exception as exc:  # noqa: BLE001 — try again at the next flush
                logger.warning("voice_usage_save_failed", error=type(exc).__name__)
        if monitor.turn_limit_reached and not monitor.agent_busy:
            await _end_call(session, "turn_limit")  # TURN_LIMIT was said instead of an answer
            return
        action = monitor.tick(now)
        if action is None:
            continue
        logger.info("voice_limit", action=action.value, elapsed_s=round(monitor.elapsed(now)))
        if action in ENDS_CALL:
            try:
                session.interrupt(force=True)  # don't finish a long answer past the limit
            except RuntimeError:
                pass  # nothing playing
            try:
                await session.say(LINES[action], allow_interruptions=False, add_to_chat_ctx=False)
            finally:
                await _end_call(session, action.value)
            return
        session.say(LINES[action], add_to_chat_ctx=False)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    if ctx.is_fake_job():
        # `console` mode: the laptop's own mic and speaker, no room and no token
        # to check. A local identity keeps its data apart from real users'.
        user_id = "g-local-console"
        thread_id = new_voice_room(user_id)
    else:
        user_id, thread_id = participant.identity, ctx.room.name
        denied = check_join(user_id, thread_id, participant.metadata)
        if denied:
            await _refuse(ctx, denied)
            return

    from src.persistence.user_data import UserData

    voice = get_settings().voice
    data = UserData(user_id)
    used_today = await asyncio.to_thread(data.voice_seconds_today)
    monitor = CallMonitor(voice, used_today, time.monotonic())
    if monitor.budget_s <= 0:
        await _refuse(ctx, "daily_limit", DAILY_LIMIT)
        return

    # Records the call without touching the chat page's active thread.
    await asyncio.to_thread(data.start_voice_call, thread_id)
    usage = _Usage(data, thread_id, monitor.started)
    agent = FinnieVoiceAgent(user_id=user_id, thread_id=thread_id)
    agent.monitor = monitor
    logger.info("voice_call_started", budget_s=round(monitor.budget_s))

    tts_kwargs: dict[str, Any] = {"model": voice.tts_model}
    if voice.tts_voice_id:
        tts_kwargs["voice_id"] = voice.tts_voice_id
    session = AgentSession(
        stt=deepgram.STT(model=voice.stt_model, language="en-US", keyterm=KEYTERMS, smart_format=True),
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
            "tts_model": voice.tts_model,
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
        # Logs never carry what the user said or the answer; the local file may.
        logger.info("voice_turn", **{k: v for k, v in record.items() if k not in ("question", "filler", "at")})
        if TURN_LOG:
            path = Path(TURN_LOG)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    @session.on("user_input_transcribed")
    def _on_transcript(event: Any) -> None:
        agent.on_transcript(event.transcript, event.is_final)

    @session.on("user_state_changed")
    def _on_user_state(event: Any) -> None:
        monitor.set_user_speaking(event.new_state == "speaking", time.monotonic())
        if event.new_state == "speaking":
            agent.on_user_speaking()

    @session.on("agent_state_changed")
    def _on_agent_state(event: Any) -> None:
        monitor.set_agent_busy(event.new_state in ("thinking", "speaking"), time.monotonic())

    tasks: dict[str, asyncio.Task] = {}

    async def wrap_up(reason: str) -> None:
        """However the call ends (hang-up, a limit, End): stop, save, close the room."""
        if (limits := tasks.get("limits")) is not None and limits is not asyncio.current_task():
            limits.cancel()
        await usage.finish(time.monotonic())
        logger.info("voice_call_ended", reason=reason, seconds=round(time.monotonic() - usage.started))
        await ctx.delete_room()
        # Without this the job lingers until LiveKit empties the room (~20 s).
        ctx.shutdown(reason=reason)

    @session.on("close")
    def _on_close(event: Any) -> None:
        reason = getattr(event, "reason", "closed")
        tasks["wrap_up"] = asyncio.ensure_future(wrap_up(str(getattr(reason, "value", reason))))

    # Backstop for a shutdown that skips the session's close (worker draining).
    ctx.add_shutdown_callback(lambda: usage.finish(time.monotonic()))

    await session.start(agent=agent, room=ctx.room)
    tasks["limits"] = asyncio.create_task(_enforce_limits(session, monitor, usage), name="voice-limits")
    await session.say(SESSION_DISCLAIMER, allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
