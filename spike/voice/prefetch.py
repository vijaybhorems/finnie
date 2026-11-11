"""Work the voice worker starts before a turn ends, and what it says while it waits.

Nothing here writes anything a user could see later: classification only fills
the classifier's short verdict cache, and warming only fills the same data
caches the agents read (5 min for prices, 30-60 min for sectors, macro and
filings). That is why it can run while the user is still speaking, unlike
LiveKit's preemptive generation, which runs the whole turn (design doc,
"Preemptive generation").
"""
from __future__ import annotations

import time
from typing import Any

# Said at once on turns that fetch data or compute before answering. Each line
# only says what Finnie is about to do, never what the answer will be.
FILLERS: dict[str, str] = {
    "market_analysis": "Let me check the latest market numbers.",
    "news_synthesizer": "Let me look at the latest news.",
    "goal_planning": "Let me work through that.",
    "portfolio": "Let me take a look.",
}


def warm_shared_data() -> dict[str, Any]:
    """Fetch the live data every turn of its kind needs, whatever the question.

    Mirrors the fixed calls in MarketAnalysisAgent (index quotes, sector sweep),
    GoalPlanningAgent (rate environment), FinanceQAAgent (macro snapshot) and
    NewsSynthesizerAgent (SEC filings) so the agent's own call is a cache hit.
    Question-specific fetches (a named ticker, headlines for the query) still
    happen in the turn. Returns how long it took and which fetches failed.
    """
    from src.agents.market_analysis_agent import MarketAnalysisAgent
    from src.data.fred_client import FredClient
    from src.data.news_client import NewsClient
    from src.data.yfinance_client import YFinanceClient
    from src.utils.parallel import gather

    yf, fred, news = YFinanceClient(), FredClient(), NewsClient()
    tasks: dict[str, Any] = {
        f"index:{t}": (lambda t=t: yf.get_current_price(t)) for t in MarketAnalysisAgent.INDEX_TICKERS
    }
    tasks["sector"] = yf.get_sector_performance
    tasks["rate_env"] = fred.get_interest_rate_environment
    tasks["macro"] = fred.get_macro_snapshot
    tasks["filings"] = lambda: news.get_sec_filings(max_items=5)

    start = time.perf_counter()
    results = gather(tasks)
    return {
        "seconds": round(time.perf_counter() - start, 3),
        "failed": sorted(k for k, v in results.items() if v is None),
    }
