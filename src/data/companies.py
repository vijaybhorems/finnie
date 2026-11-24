"""Companies the user named: symbol, name, latest price and recent news.

The classifier gives two lists: `companies`, names as the user said them
("Anthropic"), and `tickers`, the symbols it is sure of. Its symbols come from
the model's training, so a company that listed since then has none. Each name
is looked up with Yahoo Finance search instead, which knows today's listings,
and the same search returns the company's recent news in one call.
"""
from __future__ import annotations

from typing import Any

from src.data.yfinance_client import YFinanceClient
from src.utils.parallel import gather

MAX_COMPANIES = 3


def research_companies(
    names: list[str],
    tickers: list[str],
    yf: YFinanceClient,
    *,
    with_prices: bool = True,
    news_count: int = 5,
) -> list[dict[str, Any]]:
    """One entry per company: {query, symbol, name, exchange, price, news}.

    Names come first; a ticker that one of them already resolved to is not
    repeated. A name the search can't match keeps `symbol` None, so the answer
    can say nothing was found rather than the model guessing. A ticker the
    search doesn't know is kept as given.
    """
    queries = list(dict.fromkeys([n for n in names if n.strip()] + [t for t in tickers if t.strip()]))
    queries = queries[: MAX_COMPANIES * 2]
    if not queries:
        return []
    ticker_set = set(tickers)
    searched = gather({q: (lambda q=q: yf.search(q, news_count=news_count)) for q in queries})

    companies: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        result = searched.get(query) or {}
        quotes = result.get("quotes") or []
        if query in ticker_set:
            quote = next((q for q in quotes if q["symbol"] == query.upper()), None)
            symbol = query.upper()
        else:
            quote = quotes[0] if quotes else None
            symbol = quote["symbol"] if quote else None
        if symbol and symbol in seen:
            continue
        if symbol:
            seen.add(symbol)
        companies.append({
            "query": query,
            "symbol": symbol,
            "name": quote["name"] if quote else "",
            "exchange": quote["exchange"] if quote else "",
            "price": None,
            "news": (result.get("news") or [])[:news_count],
        })
        if len(companies) == MAX_COMPANIES:
            break

    if with_prices:
        symbols = [c["symbol"] for c in companies if c["symbol"]]
        prices = gather({s: (lambda s=s: yf.get_current_price(s)) for s in symbols})
        for c in companies:
            if c["symbol"]:
                c["price"] = prices.get(c["symbol"])
    return companies


def format_companies(companies: list[dict[str, Any]]) -> str:
    """The companies as a context section for the model."""
    if not companies:
        return "No specific company named."
    blocks = []
    for c in companies:
        if not c["symbol"]:
            blocks.append(f'"{c["query"]}": no listed security found in a live market search.')
            continue
        title = f'{c["name"] or c["symbol"]} ({c["symbol"]}'
        title += f', {c["exchange"]})' if c["exchange"] else ")"
        lines = [title]
        price = c.get("price") or {}
        if price.get("current_price") is not None:
            change = price.get("change_pct")
            lines.append(
                f'  Last price: {price["current_price"]:.2f} {price.get("currency") or "USD"}'
                + (f" ({change:+.2f}% vs previous close)" if change is not None else "")
            )
        elif c.get("price") is not None:
            lines.append("  Price: unavailable right now")
        if c["news"]:
            lines.append("  Recent news:")
            for n in c["news"]:
                when = f' ({n["published_at"][:10]})' if n.get("published_at") else ""
                lines.append(f'  • [{n.get("source") or "Unknown"}] {n["title"]}{when}')
        else:
            lines.append("  Recent news: none found")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def company_headlines(companies: list[dict[str, Any]]) -> list[str]:
    return [n["title"] for c in companies for n in c["news"]]
