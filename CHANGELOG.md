# Changelog

Every change merged into `main`, newest first. Dates are merge dates (UTC).

New entries go under **Unreleased** automatically when a pull request merges (`.github/workflows/changelog.yml`). To control your PR's entry, add one line to its description:

- `Changelog: Fixed: The Portfolio tab no longer draws its charts twice` (type is `Added`, `Changed` or `Fixed`)
- `Changelog: skip` for changes nobody outside the repo would notice

Without that line, the PR title is used, and the type is guessed from its first word ("Fix…" → Fixed, "Add…" → Added, anything else → Changed). When you deploy, rename **Unreleased** to the deploy date.

## Unreleased

## 2026-11

### Added
- **Voice spike** ([#36](https://github.com/vijaybhorems/finnie/pull/36)): a LiveKit voice agent (Deepgram speech-to-text, the Finnie graph, ElevenLabs speech) under `spike/voice`, with a scripted caller that measures time to first audio. Time to first audio fell from 5–9 s to about 2.5–4 s. Supporting app changes:
  - `channel="voice"`: short spoken answers, and the shared FAQ cache is skipped.
  - `llm.thinking` turns off adaptive thinking, and `llm.classifier_model` sets a separate classifier model. Both are off by default.
  - A 120 s classifier verdict cache.
  - Company names resolve to tickers ("Apple" → AAPL) for market questions, in chat too.
- **CI** ([#35](https://github.com/vijaybhorems/finnie/pull/35), first attempt [#33](https://github.com/vijaybhorems/finnie/pull/33)): GitHub Actions runs the test suite on every pull request and push to `main`, against a real Postgres + pgvector service. #33 merged into another PR's branch by mistake, and #35 re-landed it on `main`.

### Fixed
- **Chat repeated every answer while streaming** ([#34](https://github.com/vijaybhorems/finnie/pull/34)): the stream also forwarded the agent's finished message, so answers appeared two or three times until the page redrew.
- **Routing evals scored the wrong router** ([#31](https://github.com/vijaybhorems/finnie/pull/31)): they now score the merged classifier that production uses. The classifier prompt now separates market questions from news questions, which fixes the one case it missed.
- **Four stale tests** ([#32](https://github.com/vijaybhorems/finnie/pull/32)): updated sector mocks, the lazy Redis patch and feedparser entries to match the current code. Tests only.

## 2026-10

### Changed
- **Faster cold starts on Cloud Run** ([#30](https://github.com/vijaybhorems/finnie/pull/30)): an instance only opens its port once the app has warmed up, so traffic only reaches warm instances. CPU is no longer throttled while idle, and the cold start is shorter.
- **Browser tab shows "Finnie" and its logo from the first byte** ([#29](https://github.com/vijaybhorems/finnie/pull/29)), instead of "Streamlit" until the script ran.
- **Market snapshot cards labelled as index ETFs** ([#28](https://github.com/vijaybhorems/finnie/pull/28)), e.g. "SPY · S&P 500", so fund share prices no longer read as index levels.
- **README brought up to date with phases 1–5** ([#25](https://github.com/vijaybhorems/finnie/pull/25)).

### Fixed
- **Phone tab bar on Streamlit 1.64** ([#27](https://github.com/vijaybhorems/finnie/pull/27)): CSS fixed for the new wrapper, and Streamlit capped below 1.65.
- **Memory panel's delete button showed no icon** ([#26](https://github.com/vijaybhorems/finnie/pull/26)): the button was too narrow for its emoji.
- **Removed Streamlit's automatic page menu and its unauthenticated page URLs** ([#24](https://github.com/vijaybhorems/finnie/pull/24)): `src/web_app/pages/` was renamed to `views/`, so Streamlit no longer serves each tab as its own page outside the sign-in gate.

## 2026-09

### Added
- **Phase 5: memory** ([#21](https://github.com/vijaybhorems/finnie/pull/21)): Finnie remembers durable facts users state in chat ("saving for a house deposit, target 2028") and uses them in later sessions. Users can view and delete memories or switch memory off. The same PR made Finnie deployable on Google Cloud: Cloud SQL, a least-privilege runtime identity, a build-and-deploy pipeline, and migrations that run before a revision takes traffic.
- **Phase 4: knowledge base and FAQ cache on Postgres + pgvector** ([#20](https://github.com/vijaybhorems/finnie/pull/20)) when `DATABASE_URL` is set: hybrid search, and the knowledge base syncs at startup without a manual rebuild. FAISS remains the fallback.
- **Phase 3: persistence** ([#17](https://github.com/vijaybhorems/finnie/pull/17)): each signed-in user's profile, holdings, Goals plan and chat history are saved and survive refreshes and restarts.
- **Phase 2: prompt caching** ([#16](https://github.com/vijaybhorems/finnie/pull/16)): each agent's system prompt is split into stable, cached blocks, so turns stop paying full price for the same instructions.
- **Phase 1: fast path** ([#15](https://github.com/vijaybhorems/finnie/pull/15)): a semantic FAQ cache answers repeat questions with no model call. One classifier call replaces the separate guardrail and router, agents fetch data in parallel, and answers stream token by token.

### Changed
- **UI redesign, and the sign-in page renders before the heavy imports** ([#22](https://github.com/vijaybhorems/finnie/pull/22)): the model and workflow load in a background thread.
- **Mobile navigation** ([#23](https://github.com/vijaybhorems/finnie/pull/23)): a sticky tab bar and a tighter phone layout. Portfolio, Market and Goals were unreachable on phones.

### Fixed
- **Dollar amounts rendered as LaTeX** ([#19](https://github.com/vijaybhorems/finnie/pull/19)): `$` in model answers is now escaped before markdown rendering.
- **`docker compose up` from the repo root** ([#18](https://github.com/vijaybhorems/finnie/pull/18)): fixed paths that pointed at the repo's parent directory.

## 2026-07

### Added
- **Life-event planning** ([#10](https://github.com/vijaybhorems/finnie/pull/10)): a deterministic projection engine for the Goals tab that folds events (inheritance, home purchase, child birth, college, job change, retirement) onto a savings projection.
- **Tracing, guardrails and evals** ([#7](https://github.com/vijaybhorems/finnie/pull/7)): Phoenix tracing of LangChain/LangGraph, an off-topic guardrail, and routing and quality evals.
- **Circuit breakers for data APIs** ([#6](https://github.com/vijaybhorems/finnie/pull/6)): each market, macro and news client is wrapped in its own circuit breaker, with caching in front.
- **CLAUDE.md** ([#11](https://github.com/vijaybhorems/finnie/pull/11)).

### Changed
- **Model upgraded to Claude Sonnet 5, and faster page loads** ([#12](https://github.com/vijaybhorems/finnie/pull/12)): the model is configurable in `config.yaml`, and the embedding model is baked into the Docker image.
- **README updated** ([#9](https://github.com/vijaybhorems/finnie/pull/9)).

### Fixed
- **Chat errors when the model returned content blocks** ([#14](https://github.com/vijaybhorems/finnie/pull/14)): response text is normalised whether it arrives as a string or as a list of blocks.
- **Portfolio tab drew its charts twice** ([#13](https://github.com/vijaybhorems/finnie/pull/13)).
- **Phoenix trace export, prompt escaping and sector data** ([#8](https://github.com/vijaybhorems/finnie/pull/8)): spans now reach Phoenix Cloud, braces in RAG text no longer break prompts, and sector performance comes from SPDR sector ETFs, replacing Alpha Vantage's retired endpoint.

## 2026-03

### Added
- **Google OAuth sign-in** ([#4](https://github.com/vijaybhorems/finnie/pull/4)) gates every page.
- **Retries with exponential backoff** on model calls ([#2](https://github.com/vijaybhorems/finnie/pull/2)).
- **Application URL** in the README after the first Google Cloud deploy ([#5](https://github.com/vijaybhorems/finnie/pull/5)).
- **First release** (2026-03-27, committed directly): the multi-agent finance-education assistant. LangGraph routes each question to one of six agents, backed by a FAISS knowledge base, live market, macro and news data, and a Streamlit UI.

### Fixed
- **Streamlit file watcher disabled in the container** ([#1](https://github.com/vijaybhorems/finnie/pull/1)).
- **Noisy transformers warnings in the container silenced** ([#3](https://github.com/vijaybhorems/finnie/pull/3)).
