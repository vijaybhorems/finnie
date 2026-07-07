# Finnie — AI Finance Assistant 💹

A production-ready multi-agent AI system for democratizing financial education. Built with LangGraph, Claude Sonnet, FAISS, and Streamlit.

Try it out: https://finnie-app-eycr2lj5ga-uc.a.run.app/

## Architecture Overview

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  LangGraph Workflow                                                │
│                                                                    │
│  ┌────────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │ Guardrail  │───▶│  Router  │───▶│ Specialized Agent (1/6)  │  │
│  │ (finance/  │    │  Node    │    │                          │  │
│  │  NSFW gate)│    │          │    │ ┌──────────┐ ┌─────────┐ │  │
│  └─────┬──────┘    └──────────┘    │ │   RAG    │ │Data APIs│ │  │
│        │ rejected                  │ │ (FAISS)  │ │(yF/AV/  │ │  │
│        ▼                           │ └──────────┘ │FRED, w/ │ │  │
│      END (canned refusal)          │              │ circuit │ │  │
│                                     │              │ breaker)│ │  │
│                                     │              └─────────┘ │  │
│                                     └──────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
Streamlit UI (4 tabs: Chat / Portfolio / Market / Goals)
```

The guardrail is a fail-closed gate: it runs before the router on every turn, blocklist-rejects obvious NSFW/unsafe terms without an LLM call, then uses an LLM classifier for finance-scope. Any classifier error or ambiguous output also rejects — a broken guardrail can only make Finnie more restrictive, never bypass safety (`src/workflow/guardrail.py`). Each external data client (yFinance, Alpha Vantage, FRED, NewsAPI) is wrapped in its own per-provider circuit breaker (`src/utils/circuit_breaker.py`) that opens after repeated failures and self-tests via a half-open probe before closing again.

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
| LLM | Claude Sonnet 4.6 (Anthropic) |
| Orchestration | LangGraph |
| Vector DB | FAISS + sentence-transformers |
| Caching | Redis (fallback: in-memory) |
| Market Data | yFinance (free) + Alpha Vantage |
| Macro Data | FRED API |
| News | NewsAPI + RSS Feeds |
| UI | Streamlit |
| Safety Gate | LLM-based guardrail (finance/NSFW scope check, fails closed) |
| Resilience | Per-provider circuit breaker around each data client |
| Tracing/Observability | Arize Phoenix + OpenInference |
| Deployment | Docker Compose / Google Cloud Run |

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
```

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

Open http://localhost:8501 — the RAG indexer runs first, then the app starts.

### 3. Option B: Local development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build RAG index (one-time)
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

### Portfolio Tab
1. Enter holdings in the editable table (ticker, shares, avg cost)
2. Click **Fetch Current Data** to see live metrics and charts
3. Click **Get AI Analysis** for written portfolio assessment

### Market Tab
- View major index performance (SPY, QQQ, DIA, IWM)
- Sector performance heatmap (requires Alpha Vantage key)
- Historical price charts with volume
- AI market commentary

### Goals Tab
- **Projection Calculator**: Compare conservative/moderate/aggressive growth scenarios
- **AI Goal Planner**: Describe your goal in plain English, get a personalized plan
- **Retirement Calculator**: Check if you're on track for retirement

## Project Structure

```
finnie/
├── src/
│   ├── agents/              # 6 specialized agents + base class
│   │   ├── base_agent.py
│   │   ├── finance_qa_agent.py
│   │   ├── portfolio_agent.py
│   │   ├── market_analysis_agent.py
│   │   ├── goal_planning_agent.py
│   │   ├── news_synthesizer_agent.py
│   │   └── tax_education_agent.py
│   ├── core/                # Config, LLM factory, LangGraph state, tracing
│   │   ├── config.py
│   │   ├── llm.py
│   │   ├── state.py
│   │   └── tracing.py
│   ├── data/                # API clients + knowledge base
│   │   ├── yfinance_client.py
│   │   ├── alpha_vantage_client.py
│   │   ├── fred_client.py
│   │   ├── news_client.py
│   │   └── knowledge_base/  # 12 curated financial articles across 6 categories
│   ├── rag/                 # FAISS indexer + retriever
│   │   ├── indexer.py
│   │   └── retriever.py
│   ├── web_app/             # Streamlit UI (4 tabs) + Google OAuth
│   │   ├── app.py
│   │   ├── auth.py
│   │   ├── auth_bootstrap.py
│   │   └── pages/
│   │       ├── chat.py
│   │       ├── portfolio.py
│   │       ├── market.py
│   │       └── goals.py
│   ├── utils/               # Logging, Redis cache, circuit breaker
│   │   ├── cache.py
│   │   ├── circuit_breaker.py
│   │   └── logger.py
│   └── workflow/            # LangGraph graph + guardrail + router
│       ├── graph.py
│       ├── guardrail.py
│       └── router.py
├── tests/                   # pytest unit/integration suite
│   └── evals/               # LLM-as-judge + Phoenix evals (run on demand, not in CI)
├── scripts/
│   ├── build_rag_index.py
│   └── run_phoenix_evals.py # routing accuracy + answer-quality evals
├── docker/                  # Dockerfile
├── docker-compose.yml
├── config.yaml
├── requirements.txt
└── .env.example
```

## Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_agents.py -v

# Run without coverage (faster)
pytest --no-cov
```

### Evals (`tests/evals/`)

A separate suite covering RAG retrieval quality, router accuracy, guardrail behavior, resilience (circuit breaker), and disclaimer/answer-quality checks. These build a real FAISS index from the knowledge base and some cases call the live Anthropic API — run them on demand rather than in CI:

```bash
pytest tests/evals -v
```

For LLM-as-judge routing accuracy and prompt/answer-quality reports (optionally traced to Phoenix), use the standalone script instead:

```bash
python scripts/run_phoenix_evals.py --routing   # router accuracy over labelled cases
python scripts/run_phoenix_evals.py --quality   # LLM-judge answer quality (requires arize-phoenix)
python scripts/run_phoenix_evals.py --all
```

## API Documentation

### `run_workflow(user_message, conversation_history, user_profile)`

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
    }
)

print(result["final_response"])    # The agent's answer
print(result["agent_used"])        # Which agent handled the query
print(result["router_reasoning"])  # Why the router chose that agent
```

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

Add `.txt` or `.md` files to `src/data/knowledge_base/<category>/`. Then rebuild the index:

```bash
python -c "from src.rag.indexer import RAGIndexer; RAGIndexer().build_index(force=True)"
```

Categories: `investing_basics`, `portfolio_management`, `market_concepts`, `tax_accounts`, `risk_management`, `goal_planning`

## Performance Considerations

- **Caching**: Market data cached for 5 minutes; macro data for 1 hour; fundamentals for 24 hours
- **Rate limits**: Alpha Vantage free tier = 5 calls/minute. Client enforces 12s delays between calls.
- **RAG index**: Built once at startup, loaded into memory for fast retrieval (~50ms per query)
- **Redis**: Significantly improves response times for repeated queries; app falls back to in-memory cache if Redis unavailable

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

- **Guardrail** (`src/workflow/guardrail.py`): runs before the router on every turn. A fast blocklist check catches obvious NSFW/unsafe terms without an LLM call; everything else goes through an LLM topic classifier. Any error, malformed output, or off-topic verdict short-circuits the graph straight to `END` with a canned refusal — the router and agents never see a rejected query.
- **Circuit breaker** (`src/utils/circuit_breaker.py`): one breaker per data provider (yFinance, Alpha Vantage, FRED, NewsAPI). Opens after `failure_threshold` consecutive failures, stays open for `recovery_timeout_seconds`, then allows a single half-open probe request before closing again.

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

The included `docker/Dockerfile` builds directly for Cloud Run. General flow (see your specific service's current config first with `gcloud run services describe SERVICE --region=REGION --format=yaml` — image path, service account, and VPC connector for Redis all vary per deployment):

```bash
# Build & push (reuses the existing Artifact Registry repo — read the exact
# path from the describe output above rather than inventing a new one)
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/finnie-app:latest .

# Deploy a new revision — use --update-* (merge) not --set-* (replace-all),
# so you don't wipe out existing env vars/secrets/VPC config
gcloud run deploy SERVICE_NAME \
  --image=REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/finnie-app:latest \
  --region=REGION \
  --update-env-vars=TRACING_ENABLED=true,PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space \
  --update-secrets=PHOENIX_API_KEY=phoenix-api-key:latest
```

Cloud Run reaches private resources (e.g. a Memorystore Redis instance, as in `env-vars.yaml`) only via a [Serverless VPC Access connector](https://cloud.google.com/run/docs/configuring/vpc-connectors) — confirm one is attached (`vpcAccess` in the `describe` output) before assuming `REDIS_HOST` will be reachable.

## Evaluation Criteria Coverage

| Criteria | Implementation |
|----------|---------------|
| Multi-Agent Architecture (10%) | 6 agents with BaseAgent, clean separation |
| LangGraph Workflow (10%) | StateGraph with conditional routing |
| RAG Implementation (8%) | FAISS + sentence-transformers, 12 KB articles |
| Real-time Data Integration (7%) | yFinance + AV + FRED + NewsAPI with error handling |
| Streamlit Application (10%) | 4-tab UI: Chat, Portfolio, Market, Goals |
| Conversational Flow (8%) | LangGraph state, message history across turns |
| Data Visualization (7%) | Plotly charts: pie, bar, line, heatmap |
| Financial Domain Knowledge (20%) | Accurate content, proper disclaimers |
| Code Organization (5%) | Modular: agents/core/data/rag/workflow/web_app |
| Documentation (5%) | README, inline docstrings, .env.example |
| Testing (5%) | pytest with mocks, ~80%+ coverage target |

## Disclaimer

Finnie provides **financial education only** — not personalized investment advice. Always consult a licensed financial advisor before making investment decisions.
