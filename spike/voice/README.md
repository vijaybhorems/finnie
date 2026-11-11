# Voice spike

A throwaway test of the "Spike" release in the design doc *Finnie Voice Agent — Design*.
It puts LiveKit Agents in front of Finnie's existing graph:

```
browser mic ─► LiveKit room ─► worker: Deepgram STT ─► stream_workflow() ─► SpeechBuffer ─► ElevenLabs TTS ─► room ─► speaker
```

Nothing here ships. `spike/` is excluded from the Docker image, and the Talk page has no sign-in.

| File | What it does |
|---|---|
| `worker.py` | LiveKit agent. `llm_node` calls `stream_workflow()` with its own thread per session. Logs each turn to `runs/turns.jsonl`. |
| `speech.py` | Removes the written disclaimer and markdown, and releases whole sentences to TTS. Tested in `tests/test_voice_speech.py`. |
| `talk_page.py` | Local Streamlit page with Talk and End buttons and a live transcript. Mints mic-only, 10-minute tokens. |
| `measure_turns.py` | Times Finnie's share of latency (classifier, first sentence, total) with no voice vendors involved. |
| `compare_classifier.py` | Runs the production classifier over the eval suite's 44 labelled cases per model and thinking setting: routing and scope accuracy, p50/p95 latency. |
| `prefetch.py` | The live data warmed while the user speaks, and the filler lines said on slow turns. |
| `scripted_call.py` | A caller with no microphone: speaks synthesized questions into a room, interrupts on purpose, records Finnie's audio and times what a user would hear. |
| `requirements-voice.txt` | LiveKit packages, installed into a separate `.venv-voice`. |

## Setup

LiveKit Agents 1.8 needs LangGraph 1.2.20, so it gets its own environment and the main `.venv` stays untouched:

```bash
uv venv .venv-voice --python 3.12
```

```bash
uv pip install --python .venv-voice -r requirements.txt -r spike/voice/requirements-voice.txt
```

```bash
.venv-voice/bin/python -m spike.voice.worker download-files
```

Add these to `.env`. Finnie's settings ignore unknown keys.

| Key | Needed for |
|---|---|
| `DEEPGRAM_API_KEY` | Speech to text (nova-3) |
| `ELEVEN_API_KEY` | Text to speech |
| `VOICE_TTS_MODEL` | Optional. Default `eleven_flash_v2_5`; try `eleven_v3_conversational` for tone |
| `VOICE_ELEVEN_VOICE_ID` | Optional. ElevenLabs default voice if unset |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Only for LiveKit Cloud. The local dev server below needs none. |

## Run

1. Start the local LiveKit server (dev keys `devkey` / `secret`):

```bash
docker run -d --name finnie-livekit -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server:v1.13.10 --dev --bind 0.0.0.0 --node-ip 127.0.0.1
```

2. Start the worker. The first job waits about 20 s for prewarm.

```bash
.venv-voice/bin/python -m spike.voice.worker dev
```

3. Open the Talk page in a real browser (Chrome or Safari, not an embedded pane, which blocks the mic):

```bash
.venv-voice/bin/streamlit run spike/voice/talk_page.py --server.port 8520
```

To test without a browser, use the laptop mic and speaker directly:

```bash
.venv-voice/bin/python -m spike.voice.worker console
```

To run a whole call with no microphone (three questions, two interruptions; Finnie's audio lands in `runs/`):

```bash
.venv-voice/bin/python -m spike.voice.scripted_call
```

To measure Finnie alone (main venv, real Anthropic key, a few cents):

```bash
.venv/bin/python -m spike.voice.measure_turns --repeats 2
```

## Spike checklist (design doc gate)

- [x] Worker registers with LiveKit and jobs are dispatched to it
- [x] Streamlit component frames allow the microphone (`allow="microphone"` with scripts and same-origin)
- [x] Finnie's latency measured without voice vendors (below)
- [x] Speech → answer → audio end to end through the room (scripted caller, below)
- [x] Interrupting Finnie mid-answer stops it and answers the new question
- [ ] The same with a real microphone in Chrome and Safari
- [ ] Echo: on laptop speakers, Finnie doesn't hear and answer itself
- [ ] iOS Safari: the mic works over HTTPS (tunnel) and the call survives locking the screen
- [x] Time to first audio measured (below). It misses the targets.
- [ ] Cost per minute: Deepgram plus ElevenLabs usage from their dashboards after a measured session

## Findings so far (2026-11-10)

### Time to first audio, whole pipeline

From `scripted_call.py` against the local LiveKit server (Deepgram nova-3, Finnie on `claude-sonnet-5`, ElevenLabs `eleven_flash_v2_5`). "Worker" is LiveKit's per-turn metrics: turn decision + Finnie's first sentence + TTS first byte. "Heard" is the caller's own clock, from the end of its question to the first audible frame, which adds the silence the VAD waits for and transport.

| Question | Agent | Turn decision | Finnie first sentence | TTS first byte | Worker | **Heard** |
|---|---|---|---|---|---|---|
| What is a P/E ratio? | finance_qa | 0.59 s | 3.23–3.99 s | 0.23–0.31 s | 4.05–4.89 s | **5.2–5.9 s** |
| How is Apple stock doing today? | market_analysis | 0.59 s | 5.82–6.96 s | 0.20 s | 6.61–7.75 s | **8.8 s** |
| What is a P/E ratio? (repeat) | FAQ cache hit | 0.60 s | 0.01 s | 0.25 s | 0.86 s | **2.0 s** |

- **The voice plumbing alone costs about 2 s** (the FAQ hit: Finnie takes 0.01 s, the caller still waits 2.0 s). That is roughly 0.6 s turn decision, 0.25 s TTS, and about 1.1 s of VAD silence and transport. Every Finnie second comes on top.
- **Interruption works**: both interrupted answers stopped and the new question was answered. LiveKit's adaptive interruption needs LiveKit Cloud (401 locally), so the local server falls back to VAD-based interruption.
- **The Apple answer was wrong**: market_analysis said it had no live data for Apple, only the index ETFs. Spoken questions name companies ("Apple"), not tickers ("AAPL"), so ticker lookup from company names matters more for voice.
- ElevenLabs and Deepgram each stayed at 0.2–0.35 s; neither is the problem.

### Finnie's share of time to first audio

From `measure_turns.py`: real Anthropic, `claude-sonnet-5`, 2 repeats per question. The figures are time to the first speakable sentence, before any STT or TTS time.

| Turn type | First sentence p50 | p95 | Notes |
|---|---|---|---|
| FAQ cache hit | ~0.00 s | ~0.00 s | No model call |
| Q&A / tax | 2.7–2.9 s | | Classifier is about 1.55 s of this |
| Goals | 9.2–10.6 s | | Almost all of it is thinking (see below) |
| Live data (market, news) | 3.45 s | 5.91 s | Parallel fetches plus classifier |

Doc targets are ≤1.8 s p50 (Q&A) and ≤2.5 s (live data) **for the whole pipeline**. No turn type meets them yet.

### Why: adaptive thinking is on by default

`get_llm()` doesn't set `thinking`, and on Claude Sonnet 5 omitting it means **adaptive thinking**. The model reasons silently before its first text token. The goals agent, timed directly with the FRED cache warm:

| Setting | First text token | Total | Answer |
|---|---|---|---|
| Default (adaptive) | 9.83 s | 26.5 s | 4,516 chars |
| `thinking={"type": "disabled"}` | 0.52–0.59 s | 12.6–19.0 s | 3,229–4,915 chars |

This affects the chat page too: goals answers start about 9 s late there as well.

### Other findings

- **Answers are far too long for speech**: 2,400–4,700 characters, which is 1–3 minutes of audio. Voice needs its own length instruction (design doc, "Voice tone").
- **The classifier costs ~1.3–1.8 s** on Sonnet 5 before any agent starts. A faster model (Claude Haiku 5.5) or thinking off is the obvious lever.
- **Preemptive generation is on by default** in livekit-agents 1.8. The worker turns it off explicitly, as the design requires.
- **Replies need a placeholder LLM.** With no `llm=` on the session, LiveKit 1.8 silently refuses to reply even though `llm_node` is overridden. `GraphLLM` in `worker.py` is that placeholder; it is never called. Production must keep it.
- **Prewarm takes ~20 s** (embedding model plus graph). LiveKit's default `initialize_process_timeout` of 10 s killed every job, so it's raised to 120 s.
- **SpeechBuffer adds 0.03–1.0 s** over the first token, waiting for the first sentence boundary.

### Re-measured with thinking off (2026-11-10)

`llm.thinking` in `config.yaml` (default `adaptive`, which sends no parameter) can be overridden per process with `FINNIE_LLM_THINKING=disabled`. The worker sets it, so voice turns and the voice classifier run without thinking. The chat page is unchanged.

Finnie alone (`measure_turns.py`, first speakable sentence):

| Turn type | Thinking on: p50 / p95 | Thinking off: p50 / p95 |
|---|---|---|
| Q&A / tax / goals | 6.06 s / 10.64 s | **2.66 s / 3.94 s** |
| Goals only | 9.2–10.6 s | **2.5–3.9 s** |
| Live data | 3.45 s / 5.91 s | 2.90 s / 6.14 s |
| Classifier | ~1.55 s | ~1.5 s (unchanged: it wasn't thinking) |

Whole call, heard by the caller (3 calls with thinking off):

| Question | Thinking on | Thinking off |
|---|---|---|
| What is a P/E ratio? | 5.2–5.9 s | 5.3–7.9 s (no change; it barely thought) |
| How is Apple stock doing today? | 8.8 s | 7.1–7.9 s |
| How much do I need to save to retire at 55? | ~12 s (estimated from Finnie alone) | **5.7 s** |
| FAQ cache hit | 2.0 s | 1.9–2.0 s |

What is left, roughly, for a fresh Q&A turn of ~5–6 s heard: ~2 s of voice plumbing, ~1.5 s classifier, ~1.5–2.5 s for the agent's first sentence (prompt plus retrieval, memory recall and the checkpoint on a persistent thread), and data fetches on live-data turns. Run-to-run variance on the model call is about ±1.5 s.

### Classifier on Claude Haiku 5.5 (2026-11-10)

`llm.classifier_model` (env `FINNIE_CLASSIFIER_MODEL`; default unset = `llm.model`) picks the classifier's model. The worker sets `claude-haiku-5-5`; the chat page is unchanged.

`compare_classifier.py`, 32 routing + 12 scope cases from `tests/evals`:

| Config | Routing | Scope | p50 | p95 |
|---|---|---|---|---|
| Sonnet 5, thinking on (chat today) | 32/32 | 12/12 | 1.48 s | 1.78 s |
| Sonnet 5, thinking off | 32/32 | 12/12 | 1.50 s | 1.85 s |
| **Haiku 5.5, thinking off** | 32/32 | 12/12 | **1.03 s** | **1.17 s** |
| Haiku 5.5, thinking on | 32/32 | 12/12 | 1.03 s | 1.17 s |

Same accuracy, about 0.45 s faster. Thinking makes no difference for either model here: the classifier never thought. About 1 s is the floor for this call on any model. The case set is small and every configuration aced it, so it can't show subtler differences; grow it before switching the chat page.

Whole call, heard by the caller (2 calls, Haiku classifier, thinking off):

| Question | Sonnet classifier, thinking off | Haiku classifier |
|---|---|---|
| What is a P/E ratio? | 5.3–7.9 s | **4.4–4.7 s** |
| How is Apple stock doing today? | 7.1–7.9 s | 7.4–7.7 s |
| How much do I need to save to retire at 55? | 5.7 s (1 call) | 6.9–7.1 s |
| FAQ cache hit | 1.9–2.0 s | 1.8–1.9 s |

Haiku routed all six spoken questions correctly. The goals turns were slower in these calls; not the classifier. They run third, after two long answers are in the thread's history, and the goals agent always fetches FRED. Not yet confirmed.

### Short spoken answers (2026-11-10)

`stream_workflow(..., channel="voice")` (also `run_workflow`) marks a turn as spoken. `FinnieState.channel` carries it; it resets to `"chat"` every turn. A voice turn:

- gets a short-answer instruction (`_VOICE_INSTRUCTION` in `src/agents/base_agent.py`: 2–4 sentences, about 60 words, plain speech, numbers said aloud, end with an offer to go deeper). It goes in the per-request prompt block, so the cached prefix is byte-identical to chat; the live prompt-cache eval still passes.
- never reads or writes the FAQ cache. Cached answers are written for the screen, and a short spoken answer must never be served to a chat user. Repeat questions lose the ~0.9 s cache path on voice; a voice-keyed cache entry is a follow-up.

Finnie alone (`measure_turns.py --channel voice`, thinking off, Haiku classifier):

| | Chat-length answers | Voice answers |
|---|---|---|
| Answer length | 2,400–4,700 chars | **450–680 chars** (about 190 are the written disclaimer, which isn't spoken) |
| Whole turn | 10–28 s | **2.7–5 s** |
| Q&A first sentence p50 / p95 | 2.66 / 3.94 s | **2.06 / 3.83 s** |
| Live data first sentence p50 / p95 | 2.90 / 6.14 s | **2.45 / 4.08 s** |

Whole call, heard by the caller (3 calls):

| Question | Before short answers | Short answers |
|---|---|---|
| What is a P/E ratio? | 4.4–4.7 s | 3.9–4.8 s |
| How is Apple stock doing today? | 7.4–7.7 s | 6.7–8.3 s |
| How much do I need to save to retire at 55? | 6.9–7.1 s | 5.7–8.1 s |

Time to first audio barely moves: the first sentence was never the long part. What changes is everything after it. An answer now takes 15–25 s to listen to instead of 1–3 minutes, and costs a sixth of the TTS characters.

### Bug found: the stream repeated every answer (chat page too)

LangGraph's `messages` stream also emits the finished `AIMessage` an agent writes to state. It has a new id, so LangGraph doesn't know its tokens were already streamed. `stream_workflow` forwarded it, so every answer came out once as tokens, then whole, then whole again from the final flush. Voice spoke answers two or three times. The chat page showed them three times while streaming, until the rerun redrew from history. Fixed in `stream_workflow` (forward message chunks only), with a regression test in `tests/test_workflow_fast_path.py`.

### Filler line, early classification and warm data (2026-11-10)

Three changes, all in the worker (`VOICE_FILLER=0` / `VOICE_SPECULATE=0` turn them off):

- **Filler line**: once the route is known, market, news, goals and portfolio turns start with a short line that says what Finnie is about to do ("Let me check the latest market numbers."). It's followed by `FlushSentinel`. Without that, ElevenLabs' sentence splitter held the filler until the answer's first words arrived, and the filler gained nothing.
- **Early classification**: the worker classifies the transcript while the user is still speaking (interim and final transcripts, 3+ words, newest text only). `classify_query()` in `src/workflow/classify.py` keeps a 120 s per-process verdict cache keyed on the normalized text, so the turn's own classify step reuses a matching verdict. The verdict depends on the text alone, and the blocklist still runs first on every turn. It saves less than hoped: for short questions the full wording only exists at the very end, so the head start is ~0.3–0.7 s (turn's wait for a verdict: 0.4–1.1 s, was ~1.1 s). It costs about 1.6 extra Haiku calls per turn.
- **Warm data**: when the user starts talking (at most once a minute), the worker fetches the live data every turn of its kind needs: index quotes, the sector sweep, rate environment, macro snapshot and SEC filings (2.0–3.3 s cold, ~0.2 s warm). The agent's own fetch is then a cache hit. Question-specific fetches (a named ticker, headlines for the query) still happen in the turn. In production, Redis shares these caches across instances, so the gain there is mostly on quiet periods.

Nothing here saves state: classification fills the classifier's cache, and warming fills the same data caches the agents read.

Whole call, heard by the caller (3 calls; before = short answers, Haiku classifier, thinking off):

| Question | Before | Now | Worker clock now |
|---|---|---|---|
| What is a P/E ratio? (no filler) | 3.9–4.8 s | **3.7–3.9 s** | 2.6–2.8 s |
| How is Apple stock doing today? | 6.7–8.3 s | **2.5–2.9 s** | 1.4–1.8 s |
| How much do I need to save to retire at 55? | 5.7–8.1 s | **2.4–3.1 s** | 1.2–1.9 s |
| P/E again (verdict reused, no FAQ on voice) | 3.9–4.8 s | **2.9–3.3 s** | 1.9–2.2 s |

The answer itself, after the filler, now starts 0.55–1.0 s after the verdict on market and goals turns (was 4–6 s before warming), so it's ready before the filler finishes playing.

### Company names resolve to tickers (2026-11-10)

"How is Apple stock doing today?" used to get "I don't have a live quote for Apple": `MarketAnalysisAgent._extract_tickers` only matches uppercase symbols. The classifier's `Verdict` now has a `tickers` field: US symbols for every company, fund or index the user names ("Apple" → AAPL, "Nvidia and AMD" → NVDA, AMD, "the S&P 500" → SPY, "Berkshire" → BRK-B). `classify_node` cleans them (uppercase, `BRK.B` → `BRK-B`, symbol-shaped only, at most 5) into `FinnieState.query_tickers`, and the market agent fetches those first, then any symbols it finds itself. This is the same call, so it costs no extra model call, and the change applies to chat too.

- Routing unchanged: 32/32 routing and 12/12 scope on both Haiku 5.5 and Sonnet 5 (thinking off). Haiku p50 1.07 s (was 1.03 s).
- The names above, plus "Tesla" and "Microsoft", all resolved correctly. "How is the tech sector doing?" gives XLK.
- Heard by the caller: Apple 2.96 s, now with a real quote ("trading around three hundred forty eight dollars, down about six tenths of one percent"). That's a little slower than 2.5–2.9 s because the turn now fetches AAPL's price and RSI, which warming can't know in advance. The answer started 1.19 s after the verdict, still under the filler.

### Levers to try next, in order

1. Tune the end-of-turn wait (VAD silence plus the 0.5 s endpointing delay) once real-mic tests show how often people pause mid-question. It's now the largest fixed cost (~1.1 s of every "heard" number).
2. Run the classifier alongside hydrate (history and memories) instead of before it, for Q&A turns that get no filler.
3. A voice-keyed FAQ cache, to get the ~0.9 s path back for repeat questions.
4. Use `query_tickers` in the news agent, so "Microsoft earnings news" searches for MSFT.
5. Decide whether the chat page should also turn thinking off for some agents and use Haiku for the classifier. That needs answer-quality evals and a larger routing set first.
