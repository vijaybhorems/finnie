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
Set VOICE_TURN_LOG=spike/voice/runs/turns.jsonl to keep per-turn timings with the question and spoken text.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterable, Awaitable, Callable

from dotenv import load_dotenv

load_dotenv()
# Same defaults as `livekit-server --dev`. LiveKit Cloud values come from .env.
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")

from livekit.agents.worker import ServerEnvOption  # noqa: E402
from livekit.agents import (  # noqa: E402
    Agent,
    APIConnectOptions,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    cli,
    llm,
    tts,
)
from livekit.agents import room_io  # noqa: E402
from livekit.agents.types import FlushSentinel  # noqa: E402
from livekit.agents.voice.agent_session import SessionConnectOptions  # noqa: E402
from livekit.plugins import deepgram, elevenlabs, noise_cancellation, silero  # noqa: E402
from livekit.plugins.turn_detector.english import EnglishModel  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.utils.logger import get_logger, setup_logging  # noqa: E402
from src.voice.access import VOICE_UNAVAILABLE, check_join, new_voice_room  # noqa: E402
from src.voice.memory import MemorySaves, should_remember  # noqa: E402
from src.voice.limits import DAILY_LIMIT, ENDS_CALL, LINES, TURN_LIMIT, CallMonitor  # noqa: E402
from src.voice.health import DEEPGRAM, ELEVENLABS, VendorHealth  # noqa: E402
from src.voice.page_events import SPEECH_DOWN, TOPIC, answer_event, encode, notice_event, spoken_event  # noqa: E402
from src.voice.prefetch import warm_shared_data  # noqa: E402
from src.voice.speech import SESSION_DISCLAIMER  # noqa: E402
from src.voice.runtime import configure_process  # noqa: E402
from src.voice.stt import clean_transcript, deepgram_options  # noqa: E402
from src.voice.usage import TokenMeter, call_summary  # noqa: E402
from src.voice.turns import (  # noqa: E402
    END,
    HEARING_GIVE_UP,
    HEARING_TROUBLE,
    STANDALONE_LINES,
    StallWatch,
    TurnOutcome,
    heard_answer,
    preamble,
    spoken_answer,
)

logger = get_logger("finnie.voice")

# Both on by default; set to 0 to measure without them.
SPECULATE = os.environ.get("VOICE_SPECULATE", "1") == "1"  # classify + warm data while the user speaks
FILLER = os.environ.get("VOICE_FILLER", "1") == "1"  # say a short line first on slow turns
WARM_EVERY_S = 60.0  # prices cache for 5 min; re-warm at most once a minute
USAGE_FLUSH_S = 30.0  # save voice seconds this often, so a crashed call still counts
TURN_LOG = os.environ.get("VOICE_TURN_LOG", "")  # local measurement only: holds what was said


configure_process()
# JSON lines (and secret redaction) like the app: Cloud Logging parses them into
# fields, which the log-based metrics and alerts read (design doc, Monitoring).
setup_logging()


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
        self._heard_before_interrupt: str | None = None
        self.memories = MemorySaves()  # waited for when the call ends
        self.tokens = TokenMeter()  # every Claude call this call makes, for the call summary
        self.turns: list[dict[str, Any]] = []
        self.turn_no = 0  # numbers the page's answer cards; 0 is the opening disclaimer
        # Speculation state for the turn being spoken.
        self._finals: list[str] = []
        self._interim = ""
        self._spec_running = False
        self._spec_pending: str | None = None
        self._spec_done: set[tuple[str, str]] = set()
        self._last_warm = 0.0
        # The previous turn, as the graph will read it from the thread: the
        # question, and the answer once its run finished ("" until then, or if
        # it failed). The classifier needs it for follow-ups ("And in twenty
        # years?"), and the early verdict is reused only if the context matches.
        self._prev_question = ""
        self._prev_answer = ""

    def _classify_context(self) -> str:
        from src.workflow.classify import conversation_context

        return conversation_context(self._prev_question, self._prev_answer)

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
        text = clean_transcript(text)  # as llm_node will see it, so the verdict is reused
        if is_final:
            self._finals.append(text.strip())
            self._interim = ""
        else:
            self._interim = text.strip()
        so_far = " ".join(t for t in [*self._finals, self._interim] if t)
        if len(so_far.split()) < 3:
            return
        if is_final:
            # At once, not queued behind an interim's call: the turn usually
            # starts ~0.3 s after the final transcript, and the full wording of
            # a short question exists only now. classify_query shares one model
            # call between this and the turn's own classify of the same words.
            asyncio.ensure_future(self._classify_early(so_far))
        elif self._spec_running:
            self._spec_pending = so_far  # only the newest text matters
        else:
            asyncio.ensure_future(self._speculate(so_far))

    async def _speculate(self, text: str) -> None:
        """Interim transcripts: one classification at a time, the newest text next."""
        self._spec_running = True
        try:
            while text:
                await self._classify_early(text)
                text, self._spec_pending = self._spec_pending or "", None
        finally:
            self._spec_running = False

    async def _classify_early(self, text: str) -> None:
        """Classify `text` once per context; fills classify_query's verdict cache."""
        from src.workflow.classify import _normalize, classify_query, is_acknowledgement
        from src.workflow.guardrail import _matches_blocklist

        key, context = _normalize(text), self._classify_context()
        if (key, context) in self._spec_done or _matches_blocklist(text) or is_acknowledgement(text):
            return
        self._spec_done.add((key, context))
        logger.info("voice_speculative_classify", words=len(key.split()))
        try:
            await asyncio.to_thread(classify_query, text, [self.tokens], context)
        except Exception:  # noqa: BLE001 — the turn classifies again and fails closed
            pass

    # ── The turn ────────────────────────────────────────────────────────────

    async def _preamble_for(self, question: str, context: str) -> list[str]:
        """Lines to say before the answer: the filler, and the securities line."""
        from src.workflow.classify import classify_query, is_acknowledgement
        from src.workflow.guardrail import _matches_blocklist

        if _matches_blocklist(question) or is_acknowledgement(question):
            return []  # refused, or "Okay." from the graph without a classifier call
        try:
            # Usually instant: classified while the user was speaking. If not,
            # this is the turn's one classifier call; the graph reuses it.
            verdict = await asyncio.to_thread(classify_query, question, [self.tokens], context)
        except Exception:  # noqa: BLE001 — the graph classifies again and fails closed
            return []
        return preamble(verdict, filler=FILLER)

    def note_interrupted(self, spoken: str) -> None:
        """The user cut in; tell the next turn what they heard of the answer."""
        self._heard_before_interrupt = heard_answer(spoken) or None

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str | FlushSentinel]:
        question, confidence = _last_user_turn(chat_ctx)
        question = clean_transcript(question)
        # The next turn's transcript starts fresh. A pending interim is this
        # turn's words, and would be classified with the next turn's context.
        self._finals, self._interim, self._spec_pending = [], "", None
        if not question:
            return
        if self.monitor is not None:
            if self.monitor.turn_limit_reached:
                return  # the goodbye is already said; the call is ending
            if not self.monitor.start_turn():
                yield TURN_LIMIT  # instead of answering; _enforce_limits ends the call after it
                return
        heard, self._heard_before_interrupt = self._heard_before_interrupt, None
        self.turn_no += 1
        turn_no = self.turn_no
        started = time.perf_counter()
        # The verdict (for the preamble) and the graph run side by side: the
        # graph loads the thread and the user's data while the classifier
        # answers, and its own classify step waits for the same model call
        # (classify_query shares it) instead of starting after the preamble.
        verdict = asyncio.ensure_future(self._preamble_for(question, self._classify_context()))
        voice = get_settings().voice
        # Read each turn, so the call bar's switch applies from the next answer.
        speak = await asyncio.to_thread(_speak_amounts, self.user_id, self.thread_id)
        loop = asyncio.get_running_loop()
        to_page = page_sender(self.user_id)  # bound here: the graph's thread has no job context
        queue: asyncio.Queue = asyncio.Queue()
        sink: dict[str, Any] = {}
        timing = {"started": time.perf_counter()}

        # From here the graph has this question in the thread: it is the next
        # turn's previous question, with no answer until the run finishes.
        self._prev_question, self._prev_answer = question, ""

        def produce() -> None:
            from src.workflow.graph import stream_workflow

            try:
                for token in stream_workflow(
                    question, sink=sink, user_id=self.user_id, thread_id=self.thread_id,
                    channel="voice", interrupted_after=heard, speak_amounts=speak,
                    callbacks=[self.tokens],
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:  # noqa: BLE001 — stream_workflow reports its own errors
                logger.warning("voice_turn_error", error=type(exc).__name__)
                sink.setdefault("agent_used", "error")
                loop.call_soon_threadsafe(queue.put_nowait, "")
            finally:
                # The answer the thread now holds (none if the run failed), unless
                # a newer turn has started: it has its own previous question.
                if self.turn_no == turn_no and sink.get("agent_used") != "error":
                    self._prev_answer = sink.get("final_response") or ""
                loop.call_soon_threadsafe(queue.put_nowait, END)
                # The full written answer, for the page, even if the user cut the speech short.
                answer = answer_event(turn_no, sink.get("agent_used"), sink.get("final_response"))
                try:
                    asyncio.run_coroutine_threadsafe(to_page(answer), loop)
                except RuntimeError:
                    pass  # the call ended while the graph was still answering
                # The run finished, even if the user interrupted the answer:
                # learn from what they said, as the chat page does.
                if should_remember(sink.get("agent_used"), confidence, voice):
                    self.memories.add(self.user_id, question, voice.memory_min_confidence)
                elif sink.get("agent_used") not in (None, "out_of_scope", "acknowledge", "error"):
                    logger.info("voice_memory_skipped_low_confidence", confidence=confidence)

        # The graph is synchronous; run it beside the event loop. On interruption
        # or an abandoned turn this generator stops reading, and the run finishes
        # on its own: cancelling it would leave the question with no answer
        # (design doc: "When the user interrupts").
        threading.Thread(target=produce, name="finnie-turn", daemon=True).start()

        lines = await verdict
        verdict_s = round(time.perf_counter() - started, 3)
        for line in lines:
            yield line
            # Speak it now. Without this, ElevenLabs' sentence splitter holds the
            # line until the answer's first words arrive, which defeats it.
            yield FlushSentinel()

        outcome = TurnOutcome()
        said: list[str] = []
        first_piece = True
        try:
            async for spoken in spoken_answer(
                queue.get, outcome,
                notice_s=voice.slow_turn_notice_seconds,
                abandon_s=voice.slow_turn_abandon_seconds,
                failed=lambda: sink.get("agent_used") == "error",
                discreet=not speak,
            ):
                said.append(spoken)
                yield spoken
                if spoken in STANDALONE_LINES:
                    yield FlushSentinel()
                elif first_piece:
                    # The answer's first sentence, complete (SpeechBuffer releases
                    # whole sentences). The sentence splitter would hold it until
                    # the next one starts, ~0.3-0.5 s later; later ones play
                    # behind it, so only the first is pushed.
                    first_piece = False
                    yield FlushSentinel()
        finally:
            # Runs on interruption too (LiveKit closes this generator), so an
            # interrupted turn logs its own timings, not the previous turn's.
            if outcome.abandoned or outcome.failed:
                logger.warning("voice_turn_not_answered", abandoned=outcome.abandoned, failed=outcome.failed)
            self.turns.append({
                "question": question,
                "spoken": " ".join(" ".join(said).split()),  # what went to text-to-speech
                "transcript_confidence": confidence,
                "agent": sink.get("agent_used"),
                "cache_hit": sink.get("cache_hit"),
                "finnie_first_token_s": outcome.first_token_s,
                "finnie_first_sentence_s": outcome.first_spoken_s,  # first token + SpeechBuffer's wait
                "finnie_total_s": round(time.perf_counter() - timing["started"], 3),
                "finnie_completed": outcome.completed,
                "abandoned": outcome.abandoned,
                "failed": outcome.failed,
                "preamble": lines,
                "after_interruption": heard is not None,
                "speak_amounts": speak,
                "verdict_s": verdict_s,  # ~0 when classified while the user spoke
                "speculate": SPECULATE,
            })


def _speak_amounts(user_id: str, thread_id: str) -> bool:
    """The user's "speak amounts" choice for this call; discreet if it can't be read."""
    from src.persistence.user_data import UserData

    try:
        return UserData(user_id).speak_amounts(thread_id)
    except Exception as exc:  # noqa: BLE001 — discreet is the safe default
        logger.warning("voice_speak_amounts_unread", error=type(exc).__name__)
        return False


def page_sender(user_id: str) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """A sender of events to the caller's page only, bound to this job's room.

    Call it on the job's event loop: LiveKit keeps the job in a context variable,
    which a coroutine scheduled from another thread (the graph's) doesn't see.
    Sending is best effort: the call goes on without it.
    """
    from livekit.agents import get_job_context

    try:
        room = get_job_context().room
    except RuntimeError:  # no job: the text-mode tests
        room = None

    async def send(event: dict[str, Any]) -> None:
        if room is None:
            return
        try:
            await room.local_participant.send_text(encode(event), topic=TOPIC, destination_identities=[user_id])
        except Exception as exc:  # noqa: BLE001 — the page just misses this event
            logger.warning("voice_page_event_not_sent", kind=event.get("type"), error=type(exc).__name__)

    return send


async def send_to_page(user_id: str, event: dict[str, Any]) -> None:
    """Send `event` to the caller's page, from the job's event loop."""
    await page_sender(user_id)(event)


def _last_user_turn(chat_ctx: llm.ChatContext) -> tuple[str, float | None]:
    """The user's latest words, and how sure speech-to-text was of them (None: not given)."""
    for item in reversed(chat_ctx.items):
        if getattr(item, "role", None) == "user":
            return (item.text_content or "").strip(), getattr(item, "transcript_confidence", None)
    return "", None


def prewarm(proc: JobProcess) -> None:
    """Load models before taking calls, as warmup.py does for the web app.

    Runs in every new call process, beside a live call whenever it replaces an
    idle process that a call just took; src/voice/preload.py keeps it short.
    """
    started, modules = time.perf_counter(), len(sys.modules)
    preloaded = "src.voice.preload" in sys.modules  # forked from LiveKit's forkserver (Linux)
    proc.userdata["vad"] = silero.VAD.load()
    from src.web_app import warmup

    warmup.wait()
    # cpu_s: all this process's CPU so far, including re-running this module after the fork.
    logger.info(
        "voice_process_ready", seconds=round(time.perf_counter() - started, 2),
        cpu_s=round(time.process_time(), 2), modules_imported=len(sys.modules) - modules, preloaded=preloaded,
    )


# Prewarm loads the embedding model and the graph (~20-40 s cold); LiveKit's
# default 10 s initialize timeout kills the process before it can take a call.
# A draining worker's shutdown callbacks wait for memory saves; give them time.
# Sizing (voice.idle_processes): see VoiceConfig and the worker pool's memory in cloudbuild.yaml.
server = AgentServer(
    setup_fnc=prewarm,
    # Imported once in the forkserver (Linux): a replacement process then starts
    # without ~4 s of imports competing with the call that just took its place.
    preload_modules=["src.voice.preload"],
    initialize_process_timeout=120,
    shutdown_process_timeout=get_settings().voice.memory_wait_seconds + 5,
    num_idle_processes=ServerEnvOption(dev_default=0, prod_default=get_settings().voice.idle_processes),
    job_memory_warn_mb=get_settings().voice.job_memory_warn_mb,
)


_NOISE_CANCELLATION = {"bvc": noise_cancellation.BVC, "nc": noise_cancellation.NC}


def _room_options() -> room_io.RoomOptions:
    """The caller's audio cleaned before VAD and speech-to-text (voice.noise_cancellation)."""
    make = _NOISE_CANCELLATION.get(get_settings().voice.noise_cancellation)
    if make is None:
        return room_io.RoomOptions()
    return room_io.RoomOptions(audio_input=room_io.AudioInputOptions(noise_cancellation=make()))


def _voices(health: VendorHealth) -> list[tts.TTS]:
    """The main voice, then the backup from another vendor; open breakers left out."""
    voice = get_settings().voice
    voices: list[tts.TTS] = []
    if not health.is_open(ELEVENLABS):
        kwargs: dict[str, Any] = {"model": voice.tts_model}
        if voice.tts_voice_id:
            kwargs["voice_id"] = voice.tts_voice_id
        voices.append(elevenlabs.TTS(**kwargs))
    if not health.is_open(DEEPGRAM):
        voices.append(deepgram.TTS(model=voice.tts_fallback_model, mip_opt_out=True))
    return voices


def _vendor_of(source: Any = None, error: Any = None) -> str | None:
    """Which vendor a plugin, or a LiveKit STTError/TTSError (by its label), belongs to."""
    if isinstance(source, (deepgram.STT, deepgram.TTS)):
        return DEEPGRAM
    if isinstance(source, elevenlabs.TTS):
        return ELEVENLABS
    label = str(getattr(error, "label", "")).lower()
    if "deepgram" in label or getattr(error, "type", "") == "stt_error":
        return DEEPGRAM
    if "elevenlabs" in label:
        return ELEVENLABS
    return None


async def _refuse(
    ctx: JobContext, reason: str, line: str = VOICE_UNAVAILABLE, health: VendorHealth | None = None,
) -> None:
    """Say `line` and end the call. Nothing is saved for a refused call."""
    logger.warning("voice_join_refused", reason=reason)
    voices = _voices(health or VendorHealth())
    session = AgentSession(tts=voices[0]) if voices else None
    try:
        if session is not None:  # no working voice: the call just ends
            await session.start(agent=Agent(instructions="Refusing this call."), room=ctx.room)
            await session.say(line, allow_interruptions=False, add_to_chat_ctx=False)
    except Exception as exc:  # noqa: BLE001 — the call still ends, just without the line
        logger.warning("voice_refusal_line_failed", error=type(exc).__name__)
    finally:
        if session is not None:
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


# Why the worker ended a call (idle, a limit, a stall), by session. LiveKit's own
# close reason for these is just "user_initiated"; the call summary wants ours.
_END_REASONS: dict[int, str] = {}


async def _end_call(session: AgentSession, reason: str) -> None:
    """End the call from the worker's side; the session's close handler wraps up."""
    logger.info("voice_call_ending", reason=reason)
    _END_REASONS.setdefault(id(session), reason)
    await session.aclose()


def _speech_usage(session: AgentSession) -> tuple[float | None, dict[str, int]]:
    """Audio seconds sent to speech-to-text, and characters spoken by each voice."""
    stt_seconds: float | None = None
    tts_chars: dict[str, int] = {}
    try:
        for usage in session.usage.model_usage:
            kind = getattr(usage, "type", "")
            if kind == "stt_usage":
                stt_seconds = (stt_seconds or 0.0) + float(usage.audio_duration)
            elif kind == "tts_usage" and usage.characters_count:
                model = usage.model or usage.provider or "unknown"
                tts_chars[model] = tts_chars.get(model, 0) + int(usage.characters_count)
    except Exception as exc:  # noqa: BLE001 — the summary still logs without speech usage
        logger.warning("voice_usage_unread", error=type(exc).__name__)
    return stt_seconds, tts_chars


async def _say_and_end(session: AgentSession, line: str, reason: str) -> None:
    try:
        session.interrupt(force=True)  # don't finish a long answer first
    except RuntimeError:
        pass  # nothing playing
    try:
        await session.say(line, allow_interruptions=False, add_to_chat_ctx=False)
    finally:
        await _end_call(session, reason)


async def _enforce_limits(
    session: AgentSession, monitor: CallMonitor, usage: _Usage, stalls: StallWatch,
) -> None:
    """Once a second: save usage now and then, and act on the call's limits and stalls."""
    while True:
        await asyncio.sleep(1)
        now = time.monotonic()
        if now - usage.saved_until >= USAGE_FLUSH_S:
            try:
                await usage.save(now)
            except Exception as exc:  # noqa: BLE001 — try again at the next flush
                logger.warning("voice_usage_save_failed", error=type(exc).__name__)
        stall = stalls.tick(now)
        if stall is not None:
            logger.warning("voice_stt_stall", in_a_row=stalls.in_a_row)
            if stall == StallWatch.END:
                await _say_and_end(session, HEARING_GIVE_UP, "stt_stalled")
                return
            session.say(HEARING_TROUBLE, add_to_chat_ctx=False)
        if monitor.turn_limit_reached and not monitor.agent_busy:
            await _end_call(session, "turn_limit")  # TURN_LIMIT was said instead of an answer
            return
        action = monitor.tick(now)
        if action is None:
            continue
        logger.info("voice_limit", action=action.value, elapsed_s=round(monitor.elapsed(now)))
        if action in ENDS_CALL:
            await _say_and_end(session, LINES[action], action.value)
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
    health = VendorHealth()
    if await asyncio.to_thread(health.is_open, DEEPGRAM):
        # Can't hear the user: end now rather than fail mid-call.
        await _refuse(ctx, "stt_circuit_open", health=health)
        return
    voices = await asyncio.to_thread(_voices, health)
    if not voices:
        await _refuse(ctx, "tts_circuit_open", health=health)
        return
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

    # Per-call vendor bookkeeping for the shared breakers.
    failed: set[str] = set()
    worked = {"heard": False, "spoke": False}

    def vendor_failed(vendor: str | None, what: str) -> None:
        if vendor and vendor not in failed:  # once per call: retries aren't separate outages
            failed.add(vendor)
            logger.warning("voice_vendor_failed", vendor=vendor, what=what)
            asyncio.ensure_future(asyncio.to_thread(health.failure, vendor))

    speech: tts.TTS = voices[0]
    # The main voice's breaker was open, so the call starts on the backup.
    backup_used = {"yes": _vendor_of(voices[0]) != ELEVENLABS}
    if len(voices) > 1:
        # The rest of the call moves to the backup once the main voice fails.
        speech = tts.FallbackAdapter(voices, max_retry_per_tts=1)

        voices_down: set[int] = set()

        @speech.on("tts_availability_changed")
        def _on_voice_availability(event: Any) -> None:
            if event.available:
                voices_down.discard(id(event.tts))
                return
            vendor_failed(_vendor_of(event.tts), "tts_unavailable")
            voices_down.add(id(event.tts))
            if event.tts is voices[0]:
                backup_used["yes"] = True  # the rest of the call speaks with the backup
            if len(voices_down) == len(voices):  # nothing left to speak with
                asyncio.ensure_future(send_to_page(user_id, notice_event(SPEECH_DOWN)))

    stalls = StallWatch(voice.stt_stall_seconds, voice.stt_stalls_to_end)
    session = AgentSession(
        stt=deepgram.STT(**deepgram_options(voice.stt_model)),
        llm=GraphLLM(),
        tts=speech,
        vad=ctx.proc.userdata["vad"],
        conn_options=SessionConnectOptions(
            tts_conn_options=APIConnectOptions(timeout=voice.tts_connect_timeout_seconds),
            # LiveKit closes the session after this many failed reconnects in a
            # row (3 by default, ~2 s for Deepgram), before StallWatch can say
            # anything. The worker ends the call itself, after saying why.
            max_unrecoverable_errors=voice.vendor_error_tolerance,
        ),
        turn_handling={
            "turn_detection": EnglishModel(),
            "endpointing": {"min_delay": voice.endpointing_min_delay, "max_delay": voice.endpointing_max_delay},
            # "uh-huh" while Finnie speaks shouldn't stop the answer; a cut-in
            # with no words resumes it.
            "interruption": {
                # LiveKit leaves adaptive off in production mode unless it is named.
                "mode": voice.interruption_mode,
                "min_duration": voice.interruption_min_duration,
                "min_words": voice.interruption_min_words,
                "resume_false_interruption": True,
                "false_interruption_timeout": voice.false_interruption_timeout,
            },
            # On by default in livekit-agents 1.8; the design keeps it off because
            # a discarded run would still write the checkpoint, FAQ cache and memories.
            "preemptive_generation": {"enabled": False},
        },
    )

    pending_user: dict[str, Any] = {}
    interruptions = {"count": 0}

    @session.on("conversation_item_added")
    def _on_item(event: Any) -> None:
        item = event.item
        role = getattr(item, "role", None)  # handoff records have no role
        metrics = dict(getattr(item, "metrics", None) or {})
        if role == "user":
            pending_user.clear()
            pending_user.update(metrics)
            return
        if role != "assistant":
            return
        if getattr(item, "interrupted", False):
            # LiveKit keeps only what was played in the item: tell the next turn.
            agent.note_interrupted(item.text_content or "")
            interruptions["count"] += 1
        # What was actually said, beneath the written answer (turn 0: the disclaimer).
        asyncio.ensure_future(send_to_page(user_id, spoken_event(
            agent.turn_no, item.text_content or "", bool(getattr(item, "interrupted", False)),
        )))
        if not agent.turns:
            return
        record = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_id": user_id,  # an id, as on every worker line; simulated calls match turns by it
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
        logger.info("voice_turn", **{k: v for k, v in record.items() if k not in ("question", "spoken", "preamble", "at")})
        if TURN_LOG:
            path = Path(TURN_LOG)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    @session.on("user_input_transcribed")
    def _on_transcript(event: Any) -> None:
        if event.transcript.strip():
            stalls.transcript()
            worked["heard"] = True
        agent.on_transcript(event.transcript, event.is_final)

    @session.on("user_state_changed")
    def _on_user_state(event: Any) -> None:
        now = time.monotonic()
        monitor.set_user_speaking(event.new_state == "speaking", now)
        if event.new_state == "speaking":
            stalls.speech_started(now, agent_speaking=session.agent_state == "speaking")
            agent.on_user_speaking()
        elif getattr(event, "old_state", None) == "speaking":
            stalls.speech_ended(now)

    @session.on("agent_state_changed")
    def _on_agent_state(event: Any) -> None:
        monitor.set_agent_busy(event.new_state in ("thinking", "speaking"), time.monotonic())
        if event.new_state == "speaking":
            worked["spoke"] = True

    @session.on("error")
    def _on_error(event: Any) -> None:
        vendor = _vendor_of(event.source, event.error)
        vendor_failed(vendor, type(event.error).__name__)
        if getattr(event.error, "type", "") == "stt_error":
            stalls.failure(time.monotonic())

    tasks: dict[str, asyncio.Task] = {}

    async def wrap_up(reason: str) -> None:
        """However the call ends (hang-up, a limit, End): stop, save, close the room."""
        if (limits := tasks.get("limits")) is not None and limits is not asyncio.current_task():
            limits.cancel()
        await usage.finish(time.monotonic())
        await asyncio.to_thread(agent.memories.wait, get_settings().voice.memory_wait_seconds)
        seconds = time.monotonic() - usage.started
        logger.info("voice_call_ended", reason=reason, seconds=round(seconds))
        stt_seconds, tts_chars = _speech_usage(session)
        try:
            logger.info("voice_call_summary", **call_summary(
                user_id=user_id, reason=reason, seconds=seconds, turns=agent.turns,
                interruptions=interruptions["count"], stt_seconds=stt_seconds, tts_chars=tts_chars,
                tokens=agent.tokens.by_model(), vendor_failures=failed,
                backup_voice_used=backup_used["yes"], rates=get_settings().voice.cost,
            ))
        except Exception as exc:  # noqa: BLE001 — never block shutdown on monitoring
            logger.warning("voice_call_summary_failed", error=type(exc).__name__)
        # A vendor that did its job this call closes its breaker (a half-open probe passed).
        ok = {DEEPGRAM} if worked["heard"] else set()
        if worked["spoke"]:
            ok.add(_vendor_of(voices[0]) or ELEVENLABS)
        for vendor in ok - failed:
            await asyncio.to_thread(health.success, vendor)
        await ctx.delete_room()
        # Without this the job lingers until LiveKit empties the room (~20 s).
        ctx.shutdown(reason=reason)

    @session.on("close")
    def _on_close(event: Any) -> None:
        if getattr(event, "error", None) is not None:  # closed by an unrecoverable error
            vendor_failed(_vendor_of(error=event.error), type(event.error).__name__)
        reason = getattr(event, "reason", "closed")
        # The worker's own reason when it ended the call (idle, a limit, a stall).
        reason = _END_REASONS.pop(id(session), None) or str(getattr(reason, "value", reason))
        tasks["wrap_up"] = asyncio.ensure_future(wrap_up(reason))

    async def backstop() -> None:
        """A shutdown that skips the session's close (worker draining)."""
        await usage.finish(time.monotonic())
        await asyncio.to_thread(agent.memories.wait, get_settings().voice.memory_wait_seconds)

    ctx.add_shutdown_callback(backstop)

    await session.start(agent=agent, room=ctx.room, room_options=_room_options())
    tasks["limits"] = asyncio.create_task(_enforce_limits(session, monitor, usage, stalls), name="voice-limits")
    await session.say(SESSION_DISCLAIMER, allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
