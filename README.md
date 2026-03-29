# Finnie — AI Finance Assistant 💹

A production-ready multi-agent AI system for democratizing financial education. Built with LangGraph, Claude Sonnet, FAISS, and Streamlit.

Try it out: https://finnie-app-eycr2lj5ga-uc.a.run.app/

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LangGraph Workflow                                      │
│                                                          │
│  ┌──────────┐    ┌──────────────────────────────────┐  │
│  │  Router  │───▶│  Specialized Agent (one of 6)    │  │
│  │  Node    │    │                                  │  │
│  └──────────┘    │  ┌──────────┐  ┌─────────────┐  │  │
│                  │  │   RAG    │  │  Data APIs  │  │  │
│                  │  │ (FAISS)  │  │ (yF/AV/FRED)│  │  │
│                  │  └──────────┘  └─────────────┘  │  │
│                  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Streamlit UI (4 tabs: Chat / Portfolio / Market / Goals)
```

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
| Deployment | Docker Compose |

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
│   ├── core/                # Config, LLM factory, LangGraph state
│   │   ├── config.py
│   │   ├── llm.py
│   │   └── state.py
│   ├── data/                # API clients + knowledge base
│   │   ├── yfinance_client.py
│   │   ├── alpha_vantage_client.py
│   │   ├── fred_client.py
│   │   ├── news_client.py
│   │   └── knowledge_base/  # 10+ curated financial articles
│   ├── rag/                 # FAISS indexer + retriever
│   │   ├── indexer.py
│   │   └── retriever.py
│   ├── web_app/             # Streamlit UI (4 tabs)
│   │   ├── app.py
│   │   └── pages/
│   │       ├── chat.py
│   │       ├── portfolio.py
│   │       ├── market.py
│   │       └── goals.py
│   ├── utils/               # Logging, Redis cache
│   │   ├── cache.py
│   │   └── logger.py
│   └── workflow/            # LangGraph graph + router
│       ├── graph.py
│       └── router.py
├── tests/                   # pytest test suite
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

## Evaluation Criteria Coverage

| Criteria | Implementation |
|----------|---------------|
| Multi-Agent Architecture (10%) | 6 agents with BaseAgent, clean separation |
| LangGraph Workflow (10%) | StateGraph with conditional routing |
| RAG Implementation (8%) | FAISS + sentence-transformers, 10 KB articles |
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
