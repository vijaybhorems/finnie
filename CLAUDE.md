# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Finnie is a multi-agent AI finance-education assistant: LangGraph orchestrates a cache → classify → one-of-six-agents pipeline, backed by a FAISS RAG knowledge base and live market/macro/news data clients, served through a Streamlit UI behind Google OAuth.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the RAG index (required before first run; rebuild after editing src/data/knowledge_base/)
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
python scripts/run_phoenix_evals.py --routing   # LLM-judge router accuracy
python scripts/run_phoenix_evals.py --quality   # LLM-judge answer quality
python scripts/run_phoenix_evals.py --all

# Local Phoenix tracing UI (optional; tracing is off unless config.yaml/env enables it)
phoenix serve   # UI on :6006, OTLP gRPC receiver on :4317

# Docker Compose (runs the RAG indexer, then the app)
docker compose up --build
```

There is no configured linter/formatter (no ruff/black/mypy config) — `pyrightconfig.json` sets `basic` type-checking mode only.

Tests auto-mock required env vars and clear LRU-cached singletons (`get_settings`, `get_llm`, `build_graph`, circuit breakers) between tests via autouse fixtures in `tests/conftest.py` — no manual cache-clearing needed when writing new tests.

## Architecture

**Request flow:** `src/workflow/graph.py` builds a LangGraph `StateGraph` over `FinnieState` (`src/core/state.py`): `START → faq_cache → (classify → one of 6 agent nodes → faq_cache_write) | END`. `run_workflow()` is the single entry point; `stream_workflow()` is the streaming variant used by `src/web_app/pages/chat.py`.

The graph shape is controlled by two independently reversible flags under `fast_path` in `config.yaml` — with both off it rebuilds the original `START → guardrail → router → agent → END` pipeline, which is still tested:

- **FAQ cache** (`src/workflow/faq_cache.py` + `src/utils/semantic_cache.py`, flag `fast_path.faq_cache.enabled`) runs first. Exact match on the normalised query, then cosine over cached query embeddings at `similarity_threshold` (0.92). A hit ends the turn with zero LLM calls. `faq_cache_write` runs after each agent and only stores answers that are safe to reuse: `finance_qa`/`tax_education` only, `needs_macro is False`, and no live market data in `financial_data`. Every entry is stamped with `get_kb_version()` (hash of the knowledge base) and the tax year, and ignored once either moves. Every cache operation fails open — an error is a miss, never a failed turn.
- **Classify** (`src/workflow/classify.py`, flag `fast_path.merged_classifier`) replaces the guardrail and router with one `with_structured_output(Verdict)` call returning `{on_topic, agent, needs_macro, reason}`. Same safety contract as before: blocklist fast path (no LLM call), fails closed to `GuardrailConfig.refusal_message` on any error, malformed verdict, or off-topic result. `needs_macro` gates the FRED fetch in `FinanceQAAgent` — `None` means the legacy path ran and agents fetch unconditionally.
- **Legacy nodes** (`src/workflow/guardrail.py`, `router.py`) are still present and tested; `classify.py` imports the blocklist helper from `guardrail.py` rather than duplicating it.
- **Streaming** (flag `fast_path.streaming`) sets `streaming=True` on the shared `ChatAnthropic`, which is what makes LangGraph's `stream_mode="messages"` emit per-token events from `.invoke()` inside agent nodes. `stream_workflow()` filters those to `AGENT_NODES` so the classifier's own tokens never reach the UI, then flushes the disclaimer the agent appends after the model call. `get_llm(streaming=False)` is used for the structured classifier call.
- **Parallel fetches**: agents that hit more than one provider per turn (`finance_qa`, `portfolio`, `market_analysis`, `news_synthesizer`) issue them concurrently through `gather()` in `src/utils/parallel.py`; a failing task yields `None` and a warning rather than failing the turn.
- **Embeddings**: `src/core/embeddings.py` owns the one `SentenceTransformer` instance, shared by the RAG retriever/indexer and the FAQ cache — don't construct a second one.
- **Agents** (`src/agents/`) all extend `BaseAgent` (`src/agents/base_agent.py`), which owns: system-prompt assembly + disclaimer injection, and `_invoke_llm()` — a retry wrapper (3 attempts, exponential backoff) around transient `httpx` connection errors, emitting structured `llm_call_*` logs. Agents are lazily instantiated singletons per process (`_AGENTS` dict in `graph.py`), each holding its own `get_llm()` client.
  - When editing `_build_prompt()`, note that system text is escaped (`{`→`{{`) because RAG context / live data / user profile strings can contain literal braces that would otherwise be misparsed as prompt template variables.
- **State** (`FinnieState`) is the single object threaded through every node: conversation `messages` (LangGraph-managed via `add_messages`), routing fields, guardrail verdict (`is_on_topic`), `UserProfile`, `FinancialData` payload, `rag_context`, and `final_response`.
- **Config** (`src/core/config.py`): `get_settings()` is an `lru_cache`d singleton merging `config.yaml` (nested `LLMConfig`/`RAGConfig`/`CircuitBreakerConfig`/`GuardrailConfig`/`TracingConfig`/`PlanningConfig`/etc.) with env vars from `.env` (API keys, Redis host/port, Phoenix endpoint/key). If `AWS_SECRETS_NAME` is set, secrets are pulled from AWS Secrets Manager into the env *before* Settings loads (production path; local dev just uses `.env`). Env vars generally win over YAML (see Redis/Phoenix override logic at the bottom of `get_settings()`).
- **Resilience**: each external data client (`src/data/{yfinance,alpha_vantage,fred,news}_client.py`) is wrapped in its own circuit breaker (`src/utils/circuit_breaker.py`) — opens after `failure_threshold` consecutive failures, stays open `recovery_timeout_seconds`, then allows one half-open probe. Caching (`src/utils/cache.py`, Redis with in-memory fallback) sits in front of these clients: 5 min for market data, 1 hour for macro, 24 hours for fundamentals.
- **RAG** (`src/rag/`): `indexer.py` builds a FAISS index (sentence-transformers embeddings) from `src/data/knowledge_base/<category>/*.{txt,md}` (6 categories); `retriever.py` queries it. Index persists to `data/faiss_index`; load happens once at startup.
- **Planning engine** (`src/planning/`): a *pure, deterministic* projection engine separate from the LLM agents. `life_events.py` defines a closed catalog of 6 event kinds (inheritance, home purchase, child birth, college funding, job change, retirement start); `projection_engine.py` folds them year-by-year onto a baseline savings projection. With zero events it must reproduce `_project_savings` in `src/agents/goal_planning_agent.py` exactly — preserve that invariant when touching either. Defaults/caps (`default_inflation`, `max_horizon_years`, `max_events`) live under `planning` in `config.yaml`. Covered by `tests/test_life_events.py` and `tests/test_projection_engine.py`.
- **Web app** (`src/web_app/`): Streamlit, 4 tabs (Chat/Portfolio/Market/Goals under `pages/`). Every page is gated behind Google OAuth (`auth.py`); `st.user.is_logged_in` raises if OAuth isn't configured, so the app cannot render at all without it (no local-dev bypass). `auth_bootstrap.py` auto-generates `.streamlit/secrets.toml` from env vars in production when `GOOGLE_CLIENT_ID` is set.
- **Tracing** (`src/core/tracing.py`): `setup_tracing()` is a no-op unless `tracing.enabled` — instruments LangChain/LangGraph via OpenInference, exporting router decisions, guardrail checks, and full LLM prompt/response spans to a Phoenix collector. Exporter protocol is inferred from the endpoint scheme: `https://` → OTLP/HTTP (required for Cloud Run), `http://host:port` → gRPC (local dev via `phoenix serve`).

## Conventions

- Agents must not recommend specific securities as buys/sells and must append the standard disclaimer (`BaseAgent._add_disclaimer`) when discussing specific investments — this is asserted by evals in `tests/evals/test_disclaimer_evals.py`.
- New external data providers should follow the existing pattern: a dedicated client in `src/data/`, wrapped in its own circuit breaker, fronted by the Redis/in-memory cache.
- New life-event kinds go in `src/planning/life_events.py`'s closed catalog and must be folded into `projection_engine.py`'s year-by-year loop — keep the zero-events-reproduces-baseline invariant intact.
