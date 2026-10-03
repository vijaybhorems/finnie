# Finnie — AI Finance Assistant 💹

A production-ready multi-agent AI system for democratizing financial education. Built with LangGraph, Claude Sonnet 5, Postgres + pgvector (FAISS when there is no database), and Streamlit.

Try it out: https://finnie-app-eycr2lj5ga-uc.a.run.app/

## Architecture Overview

```
User query (signed in with Google)
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  LangGraph Workflow                                                      │
│                                                                          │
│  ┌───────────┐ miss ┌──────────┐   ┌────────────┐   ┌───────────────┐    │
│  │ FAQ cache │─────▶│ hydrate  │──▶│  classify  │──▶│ Agent (1 of 6)│    │
│  │ exact, or │      │ saved    │   │ guardrail +│   │ RAG · live    │    │
│  │ cos ≥ 0.92│      │ profile, │   │ router in  │   │ data · cached │    │
│  └─────┬─────┘      │ memories │   │ one call   │   │ prompt prefix │    │
│        │ hit        └──────────┘   └─────┬──────┘   └────────┬──────┘    │
│        │                                 │ off-topic         ▼           │
│        │                                 │           faq_cache_write     │
│        ▼                                 ▼                   │           │
│       END ◀──────── canned refusal ──────┘ ◀─────────────────┘           │
└──────────────────────────────────────────────────────────────────────────┘
    │                                 after the answer
    ▼                                       ▼
Streamlit UI (Chat / Portfolio /      memory extraction
Market / Goals)                       (background thread)

Postgres + pgvector: knowledge base, FAQ cache, chat threads, user data, memories
```

- **FAQ cache** (`src/workflow/faq_cache.py`): a repeated or reworded education question is answered from the cache with no LLM call. Only answers that are safe to reuse are stored: general Q&A and tax education, with no live data and no personal memories. Entries expire when the knowledge base or tax year changes.
- **classify** (`src/workflow/classify.py`): one structured-output call decides whether the question is in scope and which agent handles it. It is a fail-closed gate: a blocklist rejects obvious NSFW/unsafe terms without an LLM call, and any classifier error or malformed output also rejects. A broken gate can only make Finnie more restrictive, never bypass safety.
- **Agents**: answers stream token by token. Each agent's stable system prompt is cached on Anthropic's side, and agents that call several data providers call them concurrently.
- **Circuit breakers**: each external data client (yFinance, Alpha Vantage, FRED, NewsAPI) is wrapped in its own breaker (`src/utils/circuit_breaker.py`). It opens after repeated failures and self-tests with a half-open probe before closing again.

### Six Specialized Agents

| Agent | Data Sources | Responsibility |
|-------|-------------|----------------|
| **Finance Q&A** | FRED API + RAG KB | General financial education |
| **Portfolio Analysis** | yFinance + Alpha Vantage | Holdings metrics, diversification |
| **Market Analysis** | Alpha Vantage + yFinance | Real-time quotes, RSI, MACD, sectors |
| **Goal Planning** | FRED API | Retirement/savings projections |
| **News Synthesizer** | NewsAPI + RSS + SEC EDGAR | News contextualization |
| **Tax Education** | RAG KB + Static IRS data | Tax concepts, account types |

### Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude Sonnet 5 (Anthropic) — configurable in `config.yaml` |
| Orchestration | LangGraph (PostgresSaver checkpointer + PostgresStore) |
| Database | PostgreSQL + pgvector (Cloud SQL in production); in-memory when `DATABASE_URL` is unset |
| Retrieval | Hybrid pgvector search (cosine + full text, rank fusion); FAISS without a database. Embeddings: sentence-transformers `all-MiniLM-L6-v2` |
| Caching | Semantic FAQ cache; Anthropic prompt caching; Redis data cache (fallback: in-memory) |
| Memory | Facts users share in chat, extracted in the background and recalled in later sessions |
| Market Data | yFinance (free) + Alpha Vantage |
| Macro Data | FRED API |
| News | NewsAPI + RSS Feeds |
| UI | Streamlit, behind Google sign-in |
| Safety Gate | Merged guardrail + router classifier (finance/NSFW scope check, fails closed) |
| Resilience | Per-provider circuit breaker around each data client |
| Tracing/Observability | Arize Phoenix + OpenInference |
| Deployment | Docker Compose / Google Cloud Run + Cloud SQL |

**Model configuration:** the reasoning model is set via `llm.model` in `config.yaml` (default `claude-sonnet-5`), and is shared by the classifier, the memory extractor and all six agents. `src/core/llm.py` only sends the `temperature` sampling parameter to models that accept it — newer models (Sonnet 5, Opus 4.8/4.7) reject sampling params, so it is omitted for them automatically. Swapping to a more capable model (e.g. `claude-opus-4-8`) or a cheaper one is a one-line config change.

## Setup Instructions

### 1. Clone and configure

```bash
git clone <repo-url>
cd finnie
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
ANTHROPIC_API_KEY=sk-ant-...       # Required
ALPHA_VANTAGE_API_KEY=...          # Optional — enables technical indicators
FRED_API_KEY=...                   # Optional — enables macro data
NEWS_API_KEY=...                   # Optional — enables news headlines
DATABASE_URL=postgresql://...      # Optional — saved data, chat history, memory, pgvector retrieval
```

Without `DATABASE_URL`, Finnie runs entirely in memory: retrieval uses the FAISS index, and profiles, holdings, conversations and memories are lost on restart (a `persistence_in_memory` warning says so). Docker Compose sets `DATABASE_URL` for you.

**Free API keys:**
- Anthropic: https://console.anthropic.com
- Alpha Vantage: https://www.alphavantage.co/support/#api-key (free tier: 25 req/day)
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html (free, unlimited)
- NewsAPI: https://newsapi.org/register (free tier: 100 req/day)

**Google sign-in (required to load the app at all — there is no local-dev bypass):**

The app gates every page behind Google OAuth (`src/web_app/auth.py`). Without valid credentials configured, `st.user.is_logged_in` raises an error and the app won't render, even locally. Create a Google OAuth client at the [Google Cloud Console](https://console.cloud.google.com/apis/credentials) (type "Web application", redirect URI `http://localhost:8501/oauth2callback` for local dev), then either:

- Add the values to `.env` (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `AUTH_REDIRECT_URI`, `AUTH_COOKIE_SECRET` — generate the cookie secret with `python -c "import secrets; print(secrets.token_hex(32))"`), **or**
- Create `.streamlit/secrets.toml` directly from `.streamlit/secrets.toml.example` (this is what `auth_bootstrap.py` does automatically in production when `GOOGLE_CLIENT_ID` is set as an env var — see `src/web_app/auth_bootstrap.py`).

Leave `ALLOWED_EMAILS` empty to allow any authenticated Google account, or set a comma-separated allowlist to restrict access.

### 2. Option A: Docker Compose (recommended)

```bash
docker compose up --build
```

Open http://localhost:8501. Compose starts Redis and Postgres (pgvector) first; the app then creates its tables and loads the knowledge base into Postgres before it serves.

### 3. Option B: Local development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies (requirements-dev.txt adds the local Phoenix UI and evals;
# the container installs requirements.txt only)
pip install -r requirements-dev.txt

# Optional: Postgres with pgvector, for saved data, chat history and memory
docker run -d -p 5432:5432 -e POSTGRES_USER=finnie -e POSTGRES_PASSWORD=finnie -e POSTGRES_DB=finnie pgvector/pgvector:pg16
export DATABASE_URL=postgresql://finnie:finnie@localhost:5432/finnie
python -m src.persistence.migrate   # tables, pgvector schema, knowledge-base sync

# Without a database instead: build the FAISS index (one-time)
python -c "from src.rag.indexer import RAGIndexer; RAGIndexer().build_index()"

# Start Redis (optional, for caching)
docker run -d -p 6379:6379 redis:7-alpine

# Run the app
streamlit run src/web_app/app.py
```

## Usage Examples

### Chat Tab
Natural conversation with automatic agent routing:
- *"What is a P/E ratio?"* → Finance Q&A Agent
- *"Analyze: AAPL 10 shares @ $150, MSFT 5 @ $300"* → Portfolio Agent
- *"How do I retire at 55 with $80k/year?"* → Goal Planning Agent
- *"What's happening in the market today?"* → News Synthesizer

With a database, conversations are saved per user and survive a refresh or restart. Finnie also remembers durable facts you mention, such as *"I'm saving for a house deposit, target 2028"*, and uses them in later conversations. The sidebar's **What Finnie remembers** panel lists those facts, lets you delete any of them or forget everything, and can switch memory off. Anything that looks like an account or card number, an email or a password is filtered out before saving.

### Portfolio Tab
1. Enter holdings in the editable table (ticker, shares, avg cost). Your own holdings are saved; the example rows are not.
2. Click **Fetch Current Data** to see live metrics and charts
3. Click **Get AI Analysis** for written portfolio assessment. The chat agents see saved holdings too, so *"How diversified is my portfolio?"* works in Chat.

### Market Tab
- View major index performance (SPY, QQQ, DIA, IWM)
- Sector performance heatmap (requires Alpha Vantage key)
- Historical price charts with volume
- AI market commentary

### Goals Tab
- **Projection Calculator**: Compare conservative/moderate/aggressive growth scenarios
- **Life Timeline**: Compose a sequence of life events (home purchase, child, college funding, job change, inheritance, retirement) on top of a baseline projection and see the combined effect on net worth over time, nominal or inflation-adjusted, with an optional AI narration of the trajectory
- **AI Goal Planner**: Describe your goal in plain English, get a personalized plan
- **Retirement Calculator**: Check if you're on track for retirement

## Project Structure

```
finnie/
├── src/
│   ├── agents/              # 6 specialized agents + base class
│   │   ├── base_agent.py        # prompt assembly (3 cached/uncached blocks), retry, disclaimer
│   │   ├── prompts/             # finnie_core.md (shared by all agents) + one .md per agent
│   │   ├── finance_qa_agent.py
│   │   ├── portfolio_agent.py
│   │   ├── market_analysis_agent.py
│   │   ├── goal_planning_agent.py
│   │   ├── news_synthesizer_agent.py
│   │   └── tax_education_agent.py
│   ├── core/                # Config, LLM factory, shared embedding model, LangGraph state, tracing
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── state.py
│   │   └── tracing.py
│   ├── data/                # API clients + knowledge base
│   │   ├── yfinance_client.py
│   │   ├── alpha_vantage_client.py
│   │   ├── fred_client.py
│   │   ├── news_client.py
│   │   └── knowledge_base/  # 12 curated financial articles across 6 categories
│   ├── rag/                 # Retrieval: pgvector (hybrid) or FAISS, same interface
│   │   ├── pgvector.py          # kb_chunks schema, hybrid search
│   │   ├── sync.py              # incremental knowledge-base sync into Postgres
│   │   ├── indexer.py           # chunking (shared) + FAISS index build
│   │   ├── retriever.py         # FAISS retriever + get_retriever()
│   │   ├── digest.py            # knowledge-base digest for the shared prompt
│   │   └── version.py           # knowledge-base hash that versions FAQ cache entries
│   ├── persistence/         # Postgres/in-memory backends, per-user data, migrations
│   │   ├── backend.py
│   │   ├── identity.py          # user id from the Google account
│   │   ├── user_data.py         # UserData: the only way to read or write a user's data
│   │   └── migrate.py           # run before serving: tables, pgvector schema, KB sync
│   ├── memory/              # Long-term memory: fact extraction + save/recall
│   │   ├── extractor.py
│   │   └── service.py
│   ├── planning/            # Deterministic multi-event net-worth projection engine
│   │   ├── life_events.py       # LifeEvent schema + closed catalog (6 event kinds)
│   │   └── projection_engine.py # Year-by-year timeline projection, pure functions
│   ├── web_app/             # Streamlit UI (4 tabs) + Google OAuth
│   │   ├── app.py               # sign-in gate + tab navigation
│   │   ├── serve.py             # container entry point (starts warm-up, then Streamlit)
│   │   ├── warmup.py
│   │   ├── auth.py
│   │   ├── auth_bootstrap.py
│   │   ├── session.py           # signed-in user's id and saved-data access for pages
│   │   ├── memory_panel.py      # "What Finnie remembers" sidebar panel
│   │   ├── theme.py
│   │   └── views/               # never name this pages/ — Streamlit would serve each file ungated
│   │       ├── chat.py
│   │       ├── portfolio.py
│   │       ├── market.py
│   │       └── goals.py         # incl. Life Timeline sub-tab (src/planning)
│   ├── utils/               # Logging, caches, circuit breaker, concurrent fetches
│   │   ├── cache.py
│   │   ├── semantic_cache.py    # FAQ cache: Postgres or Redis + in-process vectors
│   │   ├── circuit_breaker.py
│   │   ├── parallel.py
│   │   └── logger.py
│   └── workflow/            # LangGraph graph and nodes
│       ├── graph.py             # build_graph, run_workflow, stream_workflow
│       ├── faq_cache.py
│       ├── hydrate.py
│       ├── classify.py          # merged guardrail + router
│       ├── guardrail.py         # legacy path (fast_path.merged_classifier: false)
│       └── router.py            # legacy path
├── tests/                   # pytest unit/integration suite
│   └── evals/               # LLM-as-judge + Phoenix evals (run on demand, not in CI)
├── scripts/
│   ├── build_rag_index.py
│   └── run_phoenix_evals.py # routing accuracy + answer-quality evals
├── deploy/setup_gcp.sh      # one-time Google Cloud provisioning wizard
├── docker/                  # Dockerfile + entrypoint.sh (migrate, then serve)
├── docker-compose.yml       # Redis + Postgres (pgvector) + app
├── cloudbuild.yaml          # build, push, deploy to Cloud Run
├── config.yaml
├── requirements.txt         # runtime dependencies (what the image installs)
├── requirements-dev.txt     # + local Phoenix UI and evals
└── .env.example
```

## Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_agents.py -v

# Run without coverage (faster)
pytest --no-cov
```

The suite always uses the in-memory backend, even if your `.env` sets `DATABASE_URL`. To run the persistence, cross-user isolation, pgvector and memory tests against real Postgres, point them at a **disposable** database (the tests truncate its tables):

```bash
FINNIE_TEST_DATABASE_URL=postgresql://user@localhost:5432/finnie_test pytest tests/test_persistence.py tests/test_rag_pgvector.py tests/test_memory.py
```

### Evals (`tests/evals/`)

A separate suite covering RAG retrieval quality, router accuracy, guardrail behavior, resilience (circuit breaker), and disclaimer/answer-quality checks. These build a real FAISS index from the knowledge base and some cases call the live Anthropic API — run them on demand rather than in CI:

```bash
pytest tests/evals -v
```

Three evals check the newer features and need opting in:

```bash
FINNIE_EVAL_LIVE=1 pytest tests/evals/test_prompt_cache_evals.py -v -s   # proves prompt caching hits (live API)
FINNIE_EVAL_LIVE=1 pytest tests/evals/test_memory_evals.py -v -s         # extraction + recall across sessions (live API, a few cents)
FINNIE_TEST_DATABASE_URL=postgresql://user@localhost:5432/finnie_test pytest tests/evals/test_rag_backend_parity.py -s   # pgvector vs FAISS retrieval quality
```

The router evals exercise the legacy `router_node`, not the merged classifier that is now the default.

For LLM-as-judge routing accuracy and prompt/answer-quality reports (optionally traced to Phoenix), use the standalone script instead:

```bash
python scripts/run_phoenix_evals.py --routing   # router accuracy over labelled cases
python scripts/run_phoenix_evals.py --quality   # LLM-judge answer quality (requires requirements-dev.txt)
python scripts/run_phoenix_evals.py --all
```

## API Documentation

### `run_workflow(user_message, conversation_history, user_profile, *, user_id, thread_id)`

Main entry point for the LangGraph workflow.

```python
from src.workflow.graph import run_workflow

result = run_workflow(
    user_message="What is a P/E ratio?",
    conversation_history=[],        # list of LangChain message objects
    user_profile={
        "risk_tolerance": "moderate",      # conservative | moderate | aggressive
        "investment_horizon": "long",       # short | medium | long
        "knowledge_level": "beginner",      # beginner | intermediate | advanced
        "portfolio": [],                    # list of {ticker, shares, avg_cost}
    },
    user_id=None,     # signed-in user; their saved profile, holdings and memories are loaded
    thread_id=None,   # persistent conversation "<user_id>:<uuid>"; another user's thread is refused
)

print(result["final_response"])    # The agent's answer
print(result["agent_used"])        # Which agent handled the query
print(result["router_reasoning"])  # Why the classifier chose that agent
print(result["cache_hit"])         # True if served from the FAQ cache (no LLM call)
```

`stream_workflow(...)` takes the same arguments and yields the answer as it is generated (agent tokens only); pass a `sink` dict to receive the full result when it finishes. The chat tab uses it.

### Individual Agents

Agents can also be invoked directly:

```python
from src.agents.portfolio_agent import PortfolioAgent
from src.core.state import FinnieState, UserProfile
from langchain_core.messages import HumanMessage

agent = PortfolioAgent()
state = FinnieState(
    messages=[HumanMessage(content="Analyze my portfolio: AAPL 10 @ 150")],
    user_profile=UserProfile(risk_tolerance="moderate"),
)
result = agent.run(state)
print(result["final_response"])
```

## Extending the Knowledge Base

Add `.txt` or `.md` files to `src/data/knowledge_base/<category>/`.

- **With a database:** the app syncs the knowledge base into Postgres at startup. To apply an edit without restarting, run the sync by hand. It embeds only new or changed chunks and removes chunks whose files are gone:

  ```bash
  python -m src.rag.sync
  ```

- **Without a database (FAISS):** rebuild the index:

  ```bash
  python -c "from src.rag.indexer import RAGIndexer; RAGIndexer().build_index(force=True)"
  ```

Either way, cached FAQ answers built from the old articles stop being served, because every entry is stamped with a hash of the knowledge base.

Categories: `investing_basics`, `portfolio_management`, `market_concepts`, `tax_accounts`, `risk_management`, `goal_planning`

## Performance Considerations

- **Startup warm-up**: the LangGraph workflow and the sentence-transformers embedding model load once per process in a background thread (`src/web_app/warmup.py`). In the container, `src/web_app/serve.py` starts it as the server boots; with plain `streamlit run`, the first page render starts it. The sign-in page imports none of it, so it paints immediately on a cold instance while the load overlaps the user's Google sign-in.
- **Lazy page imports**: `app.py` imports each tab's module only when that tab is opened, so landing on the default Chat tab doesn't pull in the other tabs' dependencies (Plotly, yFinance, etc.).
- **Offline embedding model**: the Docker image bakes in the embedding model and loads it fully offline (`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`), so model load does no network round-trip to the Hugging Face Hub.
- **FAQ cache**: a repeated or reworded education question is answered with zero LLM calls. On a miss, the merged classifier keeps a turn to two LLM calls (classifier + agent) instead of three. Tuned under `fast_path` in `config.yaml`, where each feature can be switched off independently.
- **Streaming**: agent answers stream token by token (`fast_path.streaming`).
- **Prompt caching** (`llm.prompt_caching`): each agent's system prompt is three blocks, most stable first — a shared core identical for all six agents, the agent's role, then this request's context. The first two are cached by Anthropic, so a cache hit bills them at roughly a tenth of the normal input price. Never put per-request data (dates, user ids, fetched data) in the first two blocks: nothing breaks, but every call becomes a cache write. `llm_call_success` logs cache reads and writes, and `prompt_cache_inactive` warns if nothing is cached.
- **Parallel data fetches**: agents that call several providers in one turn (Finance Q&A, Portfolio, Market, News) issue the calls concurrently (`src/utils/parallel.py`); a failing provider yields a warning, not a failed turn. Multi-ticker fetches also run concurrently via `YFinanceClient.get_current_prices`.
- **Caching**: Market data cached for 5 minutes; macro data for 1 hour; fundamentals for 24 hours. The FRED macro snapshot is fetched only when the classifier says the question needs it.
- **Rate limits**: Alpha Vantage free tier = 5 calls/minute. Client enforces 12s delays between calls.
- **Retrieval**: pgvector uses exact search, which is sub-millisecond at this corpus size with perfect recall. Consider an HNSW index past ~10k chunks. The FAISS index is baked into the image for deploys without a database.
- **Redis**: Speeds up repeated data fetches; the app falls back to an in-memory cache if Redis is unavailable

## Safety Gate & Resilience

Both are tuned in `config.yaml`:

```yaml
guardrail:
  enabled: true  # pre-router finance/NSFW gate; fails closed on classifier error

circuit_breaker:
  failure_threshold: 5        # consecutive failures before a provider's breaker opens
  recovery_timeout_seconds: 60 # time before an open breaker allows a half-open probe
  success_threshold: 1         # successful probes needed to close the breaker again
```

- **Classifier** (`src/workflow/classify.py`): runs on every turn that the FAQ cache doesn't answer. A fast blocklist check catches obvious NSFW/unsafe terms without an LLM call. Everything else gets one structured-output call that returns `{on_topic, agent, needs_macro, reason}`. Any error, malformed verdict, or off-topic verdict short-circuits the graph straight to `END` with a canned refusal, so agents never see a rejected query, and the refusal is kept in the conversation history. Setting `fast_path.merged_classifier: false` restores the original separate guardrail (`src/workflow/guardrail.py`) and router (`router.py`) nodes.
- **Circuit breaker** (`src/utils/circuit_breaker.py`): one breaker per data provider (yFinance, Alpha Vantage, FRED, NewsAPI). Opens after `failure_threshold` consecutive failures, stays open for `recovery_timeout_seconds`, then allows a single half-open probe request before closing again.

## Life-Event Timeline Projection

The Goals tab's **Life Timeline** sub-tab (`src/web_app/views/goals.py`) layers a sequence of discrete life events onto a baseline savings projection using a pure, deterministic engine (`src/planning/projection_engine.py`). With no events, it reproduces the same year-by-year math as the existing single-goal projection (`_project_savings` in `src/agents/goal_planning_agent.py`) exactly.

Each event (`src/planning/life_events.py`) resolves to per-year savings and one-time net-worth deltas that the engine folds into the projection:

| Event | Effect |
|-------|--------|
| Inheritance | One-time net-worth inflow |
| Home Purchase | One-time down payment + recurring amortized mortgage payment |
| Child Birth | Recurring dependent cost, with an optional later college cost block |
| College Funding | Standalone recurring education outflow |
| Job Change | Step change (positive or negative) to annual income |
| Retirement Start | Stops ongoing contributions, begins net drawdown (spend minus Social Security) |

Defaults and caps live under `planning` in `config.yaml`:

```yaml
planning:
  default_inflation: 0.03   # used for the real (inflation-adjusted) projection view
  max_horizon_years: 60
  max_events: 25
```

The UI resolves live inflation from FRED's 5-year expectations series where available, falling back to `default_inflation`. Covered by `tests/test_life_events.py` and `tests/test_projection_engine.py`.

## Observability — Arize Phoenix Tracing

Tracing is off by default and never required for the app to run (`setup_tracing()` in `src/core/tracing.py` is a no-op unless explicitly enabled) — it instruments LangChain/LangGraph via OpenInference and exports spans (router decisions, guardrail checks, and every LLM call with full prompt/response content) to an Arize Phoenix collector.

**Local dev — self-hosted Phoenix:**

```bash
phoenix serve   # starts UI at http://localhost:6006, OTLP gRPC receiver at :4317
```

Then in `config.yaml`:

```yaml
tracing:
  enabled: true
  project_name: "finnie"
  endpoint: "http://localhost:4317"
```

**Production / Cloud Run — Phoenix Cloud:** sign up at [app.phoenix.arize.com](https://app.phoenix.arize.com), create a space, and generate an API key from that same UI (not `app.arize.com/account/api-keys` — that's a different product). Rather than editing `config.yaml`, set these as env vars (e.g. Cloud Run `--update-env-vars` / `--update-secrets`) so tracing can be toggled per-deployment without a rebuild:

```env
TRACING_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space
PHOENIX_API_KEY=...      # store as a Secret Manager secret, not a plain env var
```

The exporter protocol is inferred automatically: `https://` endpoints use OTLP over HTTP (required for Cloud Run, which only accepts HTTPS ingress), `http://host:port` endpoints use gRPC (for local dev). Spans typically show up in the Phoenix UI within a few seconds.

## Deploying to Google Cloud Run

Finnie runs on **Cloud Run**, with **Cloud SQL for PostgreSQL** (pgvector) behind the database features: saved profiles and holdings, persistent chat history, the pgvector knowledge base, the shared FAQ cache, and long-term memory. Without a database the app still runs, in-memory, and none of that survives a restart.

```
Cloud Build ── build → push :$BUILD_ID → gcloud run deploy
                                              │
Cloud Run (finnie-app, runs as finnie-run@…) ─┼─ Secret Manager  (API keys, OAuth, DATABASE_URL)
   entrypoint: migrate → streamlit            ├─ Cloud SQL       (Unix socket /cloudsql/…)
                                              └─ VPC connector   (Memorystore Redis, optional)
```

### One-time setup

Prerequisites: `gcloud auth login`, the existing service and its secrets (API keys, `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`, `AUTH_COOKIE_SECRET`), and a Google OAuth client whose redirect URI is `https://<service-url>/oauth2callback`.

```bash
deploy/setup_gcp.sh
```

The wizard walks through seven stages, checks what already exists at each (safe to re-run), and asks before anything billed or disruptive:

1. Project, region, APIs.
2. A least-privilege runtime service account (`finnie-run`), replacing the default compute account, which usually holds Editor.
3. A Cloud SQL PostgreSQL 16 instance — `db-g1-small` by default, daily backups, deletion protection. Billed.
4. The `finnie` database and user, and a `DATABASE_URL` secret. The password is generated and goes straight into Secret Manager — never shown or written to disk.
5. Secret access for the runtime account: every secret the service already uses, plus `DATABASE_URL`.
6. Cloud Build permission to deploy revisions that run as `finnie-run`.
7. The first build and deploy.

Non-secret choices are saved to `deploy/.gcp.env` (git-ignored), including the exact deploy command.

### Deploying

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_RUNTIME_SA=finnie-run@finnie-agent.iam.gserviceaccount.com,_CLOUDSQL_INSTANCE=finnie-agent:us-central1:finnie-db .
```

`cloudbuild.yaml` builds the image, pushes it with an immutable `:$BUILD_ID` tag (plus `:latest`), and deploys that tag. `gcloud run deploy` **merges** into the existing service, so settings it doesn't name — the VPC connector, other secrets, env vars such as `ALLOWED_EMAILS` — carry over unchanged. It sets:

- `--timeout=3600` and `--session-affinity` — Streamlit keeps a websocket per session; the 300s default cuts it every five minutes.
- `--service-account=finnie-run@…` and `--add-cloudsql-instances` — Cloud SQL is reached through the built-in connector as a Unix socket at `/cloudsql/<connection-name>`.
- `--memory=4Gi --cpu=2 --cpu-boost` — the torch / sentence-transformers stack OOMs at 512Mi; peak use in production has been ~1.4 GiB. Override with `_MEMORY` / `_CPU`.
- `--min-instances=1` — one instance stays warm (idle-rate billing) so visitors never hit a ~1 min cold start. Set `_MIN_INSTANCES=0` to scale to zero instead.
- `--execution-environment=gen2` — a full Linux kernel instead of gVisor; the cold-start import of torch/LangChain is bound by file-system speed.

**Migrations run before traffic.** The container entrypoint (`docker/entrypoint.sh`) runs `python -m src.persistence.migrate` — LangGraph tables, pgvector schema, knowledge-base sync — before Streamlit listens. If the database is unreachable the new revision never becomes ready and **the previous revision keeps serving**. Several instances starting at once are serialised by an advisory lock.

**Rollback:** `gcloud run services update-traffic finnie-app --region=us-central1 --to-revisions=<previous-revision>=100`, or redeploy an earlier `:$BUILD_ID` image.

### Environment variables

Plain settings live on the service; secrets live in Secret Manager. `env-vars.yaml` is git-ignored because it holds the email allowlist — start from `env-vars.example.yaml`:

```bash
gcloud run services update finnie-app --region=us-central1 --env-vars-file=env-vars.yaml
```

- `ALLOWED_EMAILS` — comma-separated. **Empty means any Google account can sign in**, so always set it in production.
- `AUTH_REDIRECT_URI` — `https://<service-url>/oauth2callback`; must match the OAuth client.
- `REDIS_HOST` / `REDIS_PORT` — optional Memorystore via the VPC connector. The app works without Redis: the FAQ cache moves to Postgres when `DATABASE_URL` is set, and data-client caching falls back to memory.

### Optional

- **Phoenix tracing:** set `TRACING_ENABLED=true` and `PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/<space>`, and add `--update-secrets=PHOENIX_API_KEY=phoenix-api-key:latest` (see [Observability](#observability--arize-phoenix-tracing)).

## Evaluation Criteria Coverage

| Criteria | Implementation |
|----------|---------------|
| Multi-Agent Architecture (10%) | 6 agents with BaseAgent, clean separation |
| LangGraph Workflow (10%) | StateGraph with conditional routing |
| RAG Implementation (8%) | Hybrid pgvector search (cosine + full text, rank fusion) or FAISS, 12 KB articles |
| Real-time Data Integration (7%) | yFinance + AV + FRED + NewsAPI with error handling |
| Streamlit Application (10%) | 4-tab UI: Chat, Portfolio, Market, Goals |
| Conversational Flow (8%) | Persistent LangGraph threads per user, plus long-term memory across sessions |
| Data Visualization (7%) | Plotly charts: pie, bar, line, heatmap |
| Financial Domain Knowledge (20%) | Accurate content, proper disclaimers |
| Code Organization (5%) | Modular: agents/core/data/rag/workflow/web_app |
| Documentation (5%) | README, inline docstrings, .env.example |
| Testing (5%) | pytest with mocks, ~80%+ coverage target |

## Disclaimer

Finnie provides **financial education only** — not personalized investment advice. Always consult a licensed financial advisor before making investment decisions.
