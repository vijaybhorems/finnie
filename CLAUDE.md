# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Finnie is a multi-agent AI finance-education assistant: LangGraph orchestrates a cache → classify → one-of-six-agents pipeline, backed by a FAISS RAG knowledge base and live market/macro/news data clients, served through a Streamlit UI behind Google OAuth.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Knowledge base — pgvector (when DATABASE_URL is set): synced automatically at app startup;
# run by hand to apply an article edit without restarting. Idempotent, embeds only changed chunks.
python -m src.rag.sync
# Database (with DATABASE_URL): LangGraph tables + pgvector schema + KB sync; the container runs this before serving
python -m src.persistence.migrate
# Knowledge base — FAISS (no DATABASE_URL): build before first run, force-rebuild after editing src/data/knowledge_base/
python -c "from src.rag.indexer import RAGIndexer; RAGIndexer().build_index()"
python -c "from src.rag.indexer import RAGIndexer; RAGIndexer().build_index(force=True)"  # force rebuild

# Run the app (requires Google OAuth configured — see below; no local-dev bypass)
streamlit run src/web_app/app.py

# Tests
pytest                                # full suite with coverage (see pytest.ini: --cov=src, html to htmlcov/)
pytest tests/test_agents.py -v        # single file
pytest tests/test_agents.py::TestClass::test_name -v   # single test
pytest --no-cov                       # faster, no coverage

# Evals — NOT run in CI; build a real FAISS index and some hit the live Anthropic API
pytest tests/evals -v
FINNIE_EVAL_LIVE=1 pytest tests/evals/test_memory_evals.py -v -s          # real extraction + recall across sessions (a few cents)
FINNIE_EVAL_LIVE=1 pytest tests/evals/test_prompt_cache_evals.py -v -s   # proves prompt caching hits (reads real key from .env)
FINNIE_TEST_DATABASE_URL=postgresql://user@localhost:5432/finnie_test pytest tests/test_persistence.py tests/test_rag_pgvector.py tests/test_memory.py   # persistence, isolation, pgvector on real Postgres (a DISPOSABLE db: tests truncate tables) (memory-only otherwise)
FINNIE_TEST_DATABASE_URL=postgresql://user@localhost:5432/finnie_test pytest tests/evals/test_rag_backend_parity.py -s   # pgvector vs FAISS retrieval quality (real model)
python scripts/run_phoenix_evals.py --routing   # LLM-judge router accuracy
python scripts/run_phoenix_evals.py --quality   # LLM-judge answer quality
python scripts/run_phoenix_evals.py --all

# Local Phoenix tracing UI (optional; tracing is off unless config.yaml/env enables it)
phoenix serve   # UI on :6006, OTLP gRPC receiver on :4317

# Google Cloud — one-time provisioning wizard (Cloud SQL, runtime SA, secrets, permissions), then build + deploy
deploy/setup_gcp.sh
gcloud builds submit --config cloudbuild.yaml --substitutions=_RUNTIME_SA=...,_CLOUDSQL_INSTANCE=...   # exact command saved in deploy/.gcp.env

# Docker Compose (Redis, Postgres+pgvector, then the app — which syncs the knowledge base at startup)
docker compose up --build
```

There is no configured linter/formatter (no ruff/black/mypy config) — `pyrightconfig.json` sets `basic` type-checking mode only.

Tests auto-mock required env vars and clear LRU-cached singletons (`get_settings`, `get_llm`, `build_graph`, circuit breakers, persistence backends) between tests via autouse fixtures in `tests/conftest.py` — no manual cache-clearing needed when writing new tests. `conftest.py` sets `DATABASE_URL=""` so the suite always uses the in-memory backend, even when a developer's `.env` points at a real database, and swaps the store's embedding function for a deterministic `hash_embed` (every turn searches memories; without it each test would load the real model). Mark a test `@pytest.mark.real_embeddings` to opt out.

## Architecture

**Request flow:** `src/workflow/graph.py` builds a LangGraph `StateGraph` over `FinnieState` (`src/core/state.py`): `START → faq_cache → (hydrate → classify → one of 6 agent nodes → faq_cache_write) | END`. `run_workflow()` is the single entry point; `stream_workflow()` is the streaming variant used by `src/web_app/pages/chat.py`. Both take `user_id` (signed-in user) and `thread_id` (a persistent conversation); see Persistence below.

The graph shape is controlled by two independently reversible flags under `fast_path` in `config.yaml` — with both off it rebuilds the original `START → guardrail → router → agent → END` pipeline, which is still tested:

- **FAQ cache** (`src/workflow/faq_cache.py` + `src/utils/semantic_cache.py`, flag `fast_path.faq_cache.enabled`) runs first. Exact match on the normalised query, then cosine over cached query embeddings at `similarity_threshold` (0.92). A hit ends the turn with zero LLM calls. `faq_cache_write` runs after each agent and only stores answers that are safe to reuse: `finance_qa`/`tax_education` only, `needs_macro is False`, and no live market data in `financial_data`. Every entry is stamped with `get_kb_version()` (hash of the knowledge base) and the tax year, and ignored once either moves. Every cache operation fails open — an error is a miss, never a failed turn. Storage (`fast_path.faq_cache.backend`): `PostgresFAQCache` — a `faq_cache` table shared by every instance, validity enforced in SQL, exact matches answered without embedding the question — when `DATABASE_URL` is set; otherwise `SemanticFAQCache` on Redis with a per-process vector index.
- **Classify** (`src/workflow/classify.py`, flag `fast_path.merged_classifier`) replaces the guardrail and router with one `with_structured_output(Verdict)` call returning `{on_topic, agent, needs_macro, reason}`. Same safety contract as before: blocklist fast path (no LLM call), fails closed to `GuardrailConfig.refusal_message` on any error, malformed verdict, or off-topic result. `needs_macro` gates the FRED fetch in `FinanceQAAgent` — `None` means the legacy path ran and agents fetch unconditionally.
- **Legacy nodes** (`src/workflow/guardrail.py`, `router.py`) are still present and tested; `classify.py` imports the blocklist helper from `guardrail.py` rather than duplicating it.
- **Streaming** (flag `fast_path.streaming`) sets `streaming=True` on the shared `ChatAnthropic`, which is what makes LangGraph's `stream_mode="messages"` emit per-token events from `.invoke()` inside agent nodes. `stream_workflow()` filters those to `AGENT_NODES` so the classifier's own tokens never reach the UI, then flushes the disclaimer the agent appends after the model call. `get_llm(streaming=False)` is used for the structured classifier call.
- **Parallel fetches**: agents that hit more than one provider per turn (`finance_qa`, `portfolio`, `market_analysis`, `news_synthesizer`) issue them concurrently through `gather()` in `src/utils/parallel.py`; a failing task yields `None` and a warning rather than failing the turn.
- **Embeddings**: `src/core/embeddings.py` owns the one `SentenceTransformer` instance, shared by the RAG retriever/indexer and the FAQ cache — don't construct a second one.
- **Agents** (`src/agents/`) all extend `BaseAgent` (`src/agents/base_agent.py`), which owns: system-prompt assembly + disclaimer injection, and `_invoke_llm(state, context)` — a retry wrapper (3 attempts, exponential backoff) around transient `httpx` connection errors, emitting structured `llm_call_*` logs that include the prompt-cache usage fields. Agents are lazily instantiated singletons per process (`_AGENTS` dict in `graph.py`), each holding its own `get_llm()` client.
- **Prompts and prompt caching** (`src/agents/prompts/`, flag `llm.prompt_caching`): each agent's system prompt is three content blocks, most stable first, because Anthropic caching is a prefix match — (1) `shared_system_prompt()`: `finnie_core.md` + a knowledge-base digest (`src/rag/digest.py`), byte-identical across all six agents so one cache entry serves them all; (2) the agent's role block: name, description, `<agent>.md`, and `_static_reference()` data such as the tax tables; (3) "Context for this request": the `context` string passed to `_invoke_llm` — profile, retrieved passages, live data. Blocks 1–2 carry `cache_control`; block 3 does not.
  - **Never put per-request data in blocks 1–2** (a date, user id, query, or anything fetched per call). It still works — every request just becomes a cache write at 1.25× instead of a read at 0.1×, and nothing errors. Pass it through `context` instead.
  - Claude Sonnet 5 won't cache a prefix under 1,024 tokens; block 1 is sized to clear that alone. Shortening `finnie_core.md` or the knowledge base can drop it below — `prompt_cache_inactive` is logged once per agent when a call reads and writes nothing.
  - Prompts are sent as message objects, not a `ChatPromptTemplate`, so braces in RAG text or JSON data need no escaping.
  - After changing prompt assembly, run `FINNIE_EVAL_LIVE=1 pytest tests/evals/test_prompt_cache_evals.py -v -s` — the usage fields are the only proof the cache still hits.
- **Persistence** (`src/persistence/`, `persistence` in `config.yaml`): per-user profile, holdings, saved Goals plan, and chat history. Postgres when `DATABASE_URL` is set (LangGraph `PostgresSaver` for threads + `PostgresStore` for user data, one pool); in-process memory otherwise — nothing survives a restart there, and a `persistence_in_memory` warning says so. `build_graph(persistent=True)` adds the checkpointer (chat page); `build_graph()` stays stateless (one-shot tabs, evals). Both get the store.
  - **Identity**: `user_id` is `g-<Google sub>` from `st.user` (`src/web_app/session.py`, recomputed every call — never cached, never taken from page input). **All** user data goes through `UserData(user_id)`, which is bound to that one namespace and has no way to address another; thread ids are minted as `<user_id>:<uuid>` and `run_workflow`/`load_conversation`/`delete_conversation` refuse a thread whose prefix isn't the caller's. Isolation is application-level by design — LangGraph owns the tables and queries, so Postgres RLS has no per-request hook. `tests/test_persistence.py` is the cross-user leakage suite; keep it passing on Postgres.
  - **hydrate node** (`src/workflow/hydrate.py`) loads the user's saved data into `state.user_profile` after an FAQ-cache miss. It reads the store through LangGraph's injected `store` parameter, which is only injected for the annotation spelling `Optional[BaseStore]`/`BaseStore` — `BaseStore | None` silently disables it and every turn loses personal context with no error (`TestHydrate` catches this).
  - **Per-turn reset**: on a checkpointed thread every `FinnieState` field except `messages` carries into the next turn. `turn_state_reset()` (built from the model's defaults) is applied to each turn's input — add new fields to `FinnieState` with sensible defaults and they are reset automatically; never compute per-turn values that rely on the previous turn's state.
  - Agents send at most `workflow.max_history_messages` of history (`trim_history`, window always opens on a user turn); the full thread stays in the checkpoint. Guardrail refusals are appended to `messages` so a refused question is never left unanswered in history.
  - Pages save only on a real change or an explicit button (`save_if_changed`) — the Portfolio tab's example holdings and the Goals timeline's defaults are never saved as a user's own. Holdings are sanitised on write (`clean_holdings`): Postgres `jsonb` rejects the `NaN` that `st.data_editor` returns for blank cells.
- **Memory** (`src/memory/`, `memory` in `config.yaml`): durable facts users state in chat ("saving for a house deposit, target 2028"), recalled in later sessions.
  - **Write path**: after each on-topic chat turn, `chat.py` calls `schedule_remember_turn()` — a background thread, so the answer never waits. `extract_facts()` skips the model entirely unless the message has a first-person word, asks for structured facts (goal/situation/preference/constraint + confidence), and `remember_turn()` keeps only confidence ≥ `min_confidence` and drops anything `is_sensitive()` flags (account/card/SSN-like digit runs, emails, passwords) — a deterministic backstop to the prompt's exclusions. A fact nearly identical (`dedup_similarity`) to an existing one in the same category replaces it.
  - **Storage**: `UserData.remember/forget/forget_all/list_memories/relevant_memories`, one store item per fact in `("users", <id>, "memories")`, semantically indexed on `fact` with exact (`flat`) search. Only memory items are indexed — other `UserData` writes pass `index=False` and never embed. Deletes are real store deletes (the vector row cascades).
  - **Read path**: hydrate adds the `top_k` memories relevant to the question to `user_profile.memories`; `BaseAgent._get_user_context_str` puts them, dated, in the per-request context block (never the cached blocks).
  - **Privacy rules — keep them**: `faq_cache_write` never stores an answer produced with memories, because the FAQ cache is shared by every user. The user's switch (`memory_enabled`) stops both saving and recall. The sidebar panel (`src/web_app/memory_panel.py`) lists, deletes, and forgets everything.
- **Deployment** (Cloud Run + Cloud SQL; README "Deploying to Google Cloud Run"): `docker/entrypoint.sh` runs `python -m src.persistence.migrate` before Streamlit when `DATABASE_URL` is set, so a revision with a broken database never takes traffic. `migrate` takes its lock with a polling `pg_try_advisory_lock`, **never a blocking `pg_advisory_lock`**: LangGraph's store setup runs `CREATE INDEX CONCURRENTLY`, which waits for every open transaction, so a waiter blocked inside a lock statement deadlocks it (reproduced with 4 concurrent migrations on a fresh database). `cloudbuild.yaml` deploys an immutable `:$BUILD_ID` image with `gcloud run deploy`, which merges into the live service — unnamed settings (VPC connector, env vars, other secrets) carry over. `env-vars.yaml` is git-ignored (email allowlist); an empty `ALLOWED_EMAILS` lets any Google account in.
- **State** (`FinnieState`) is the single object threaded through every node: conversation `messages` (LangGraph-managed via `add_messages`), routing fields, guardrail verdict (`is_on_topic`), `UserProfile`, `FinancialData` payload, `rag_context`, and `final_response`.
- **Config** (`src/core/config.py`): `get_settings()` is an `lru_cache`d singleton merging `config.yaml` (nested `LLMConfig`/`RAGConfig`/`CircuitBreakerConfig`/`GuardrailConfig`/`TracingConfig`/`PlanningConfig`/etc.) with env vars from `.env` (API keys, Redis host/port, Phoenix endpoint/key). If `AWS_SECRETS_NAME` is set, secrets are pulled from AWS Secrets Manager into the env *before* Settings loads (production path; local dev just uses `.env`). Env vars generally win over YAML (see Redis/Phoenix override logic at the bottom of `get_settings()`).
- **Resilience**: each external data client (`src/data/{yfinance,alpha_vantage,fred,news}_client.py`) is wrapped in its own circuit breaker (`src/utils/circuit_breaker.py`) — opens after `failure_threshold` consecutive failures, stays open `recovery_timeout_seconds`, then allows one half-open probe. Caching (`src/utils/cache.py`, Redis with in-memory fallback) sits in front of these clients: 5 min for market data, 1 hour for macro, 24 hours for fundamentals.
- **RAG** (`src/rag/`, `rag.backend`): two interchangeable retrievers behind `get_retriever()`, same `search()`/`get_context()` interface and result shape. `auto` picks pgvector when `DATABASE_URL` is set, else FAISS — the suite and any deploy without a database use FAISS.
  - **pgvector** (`pgvector.py`): `kb_chunks` table; hybrid search — exact cosine and a Postgres full-text OR-query, each contributing `hybrid_candidates`, fused by reciprocal rank fusion (`score` is the RRF score, `similarity` the cosine). Exact search is deliberate, not an ANN index: at tens-to-thousands of chunks it is sub-millisecond with perfect recall and avoids HNSW's filtered-search recall loss; add HNSW past ~10k rows and measure. `sync_knowledge_base()` runs at startup (`warm_up`) and via `python -m src.rag.sync`: it embeds only new/changed chunks (content hash + model name), deletes chunks whose files are gone, never touches rows whose `origin` isn't `'repo'`, and is serialised by an advisory lock so instances starting together don't race. `ensure_schema()` creates the extension and tables and refuses to run if `embeddings.dimension` no longer matches the column.
  - **FAISS** (`indexer.py` + `retriever.py`): builds from `src/data/knowledge_base/<category>/*.{txt,md}` (6 categories) into `data/faiss_index` — baked into the Docker image, so a deploy without a database still has retrieval. Both backends chunk through the same `load_documents()`/`chunk_documents()`, so chunk ids and text are identical — `tests/evals/test_rag_backend_parity.py` compares them on the real corpus (at last measure: MRR@5 0.950 pgvector vs 0.867 FAISS, equal hit@5).
- **Planning engine** (`src/planning/`): a *pure, deterministic* projection engine separate from the LLM agents. `life_events.py` defines a closed catalog of 6 event kinds (inheritance, home purchase, child birth, college funding, job change, retirement start); `projection_engine.py` folds them year-by-year onto a baseline savings projection. With zero events it must reproduce `_project_savings` in `src/agents/goal_planning_agent.py` exactly — preserve that invariant when touching either. Defaults/caps (`default_inflation`, `max_horizon_years`, `max_events`) live under `planning` in `config.yaml`. Covered by `tests/test_life_events.py` and `tests/test_projection_engine.py`.
- **Web app** (`src/web_app/`): Streamlit, 4 tabs (Chat/Portfolio/Market/Goals under `pages/`). Every page is gated behind Google OAuth (`auth.py`); `st.user.is_logged_in` raises if OAuth isn't configured, so the app cannot render at all without it (no local-dev bypass). `auth_bootstrap.py` auto-generates `.streamlit/secrets.toml` from env vars in production when `GOOGLE_CLIENT_ID` is set. Load order matters for page speed: `app.py` imports nothing LangChain-related before the auth gate, so the sign-in page paints at once; the workflow and embedding model load in a background thread (`warmup.py`, started at boot by `serve.py`, the container's entry point), and signed-in pages `warmup.wait()` for it. Keep new heavy imports behind the gate. Styling lives in `.streamlit/config.toml` (`[theme]`) plus `theme.py` (CSS, sign-in page, `page_header()`).
- **Tracing** (`src/core/tracing.py`): `setup_tracing()` is a no-op unless `tracing.enabled` — instruments LangChain/LangGraph via OpenInference, exporting router decisions, guardrail checks, and full LLM prompt/response spans to a Phoenix collector. Exporter protocol is inferred from the endpoint scheme: `https://` → OTLP/HTTP (required for Cloud Run), `http://host:port` → gRPC (local dev via `phoenix serve`).

## Conventions

- Agents must not recommend specific securities as buys/sells and must append the standard disclaimer (`BaseAgent._add_disclaimer`) when discussing specific investments — this is asserted by evals in `tests/evals/test_disclaimer_evals.py`.
- New external data providers should follow the existing pattern: a dedicated client in `src/data/`, wrapped in its own circuit breaker, fronted by the Redis/in-memory cache.
- New life-event kinds go in `src/planning/life_events.py`'s closed catalog and must be folded into `projection_engine.py`'s year-by-year loop — keep the zero-events-reproduces-baseline invariant intact.
