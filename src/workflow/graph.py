"""LangGraph workflow — wires router + 6 agents into a state graph."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.goal_planning_agent import GoalPlanningAgent
from src.agents.market_analysis_agent import MarketAnalysisAgent
from src.agents.news_synthesizer_agent import NewsSynthesizerAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.tax_education_agent import TaxEducationAgent
from src.core.config import get_settings
from src.core.llm import message_text
from src.core.state import AgentType, FinnieState, UserProfile, turn_state_reset
from src.persistence.backend import get_checkpointer, get_store
from src.utils.logger import get_logger
from src.workflow.classify import classify_node, route_after_classify
from src.workflow.faq_cache import (
    faq_cache_node,
    faq_cache_write_node,
    route_after_faq_cache,
)
from src.workflow.guardrail import guardrail_node, route_after_guardrail
from src.workflow.hydrate import hydrate_node
from src.workflow.router import router_node

logger = get_logger(__name__)

# ── Agent singletons (lazy-initialised per process) ─────────────────────────
_AGENTS: dict[AgentType, Any] = {}


def _get_agent(agent_type: AgentType):
    if agent_type not in _AGENTS:
        factories = {
            AgentType.FINANCE_QA: FinanceQAAgent,
            AgentType.PORTFOLIO: PortfolioAgent,
            AgentType.MARKET_ANALYSIS: MarketAnalysisAgent,
            AgentType.GOAL_PLANNING: GoalPlanningAgent,
            AgentType.NEWS_SYNTHESIZER: NewsSynthesizerAgent,
            AgentType.TAX_EDUCATION: TaxEducationAgent,
        }
        _AGENTS[agent_type] = factories[agent_type]()
    return _AGENTS[agent_type]


# ── Node wrappers ────────────────────────────────────────────────────────────

def finance_qa_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.FINANCE_QA).run(state)


def portfolio_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.PORTFOLIO).run(state)


def market_analysis_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.MARKET_ANALYSIS).run(state)


def goal_planning_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.GOAL_PLANNING).run(state)


def news_synthesizer_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.NEWS_SYNTHESIZER).run(state)


def tax_education_node(state: FinnieState) -> dict:
    return _get_agent(AgentType.TAX_EDUCATION).run(state)


# ── Routing logic ─────────────────────────────────────────────────────────────

# Agent display name (AIMessage.name) -> node key, for redrawing saved chats.
AGENT_NAMES: dict[str, str] = {
    FinanceQAAgent.name: "finance_qa",
    PortfolioAgent.name: "portfolio",
    MarketAnalysisAgent.name: "market_analysis",
    GoalPlanningAgent.name: "goal_planning",
    NewsSynthesizerAgent.name: "news_synthesizer",
    TaxEducationAgent.name: "tax_education",
}

AGENT_NODES = [
    "finance_qa",
    "portfolio",
    "market_analysis",
    "goal_planning",
    "news_synthesizer",
    "tax_education",
]

_AGENT_NODE_MAP = {
    AgentType.FINANCE_QA: "finance_qa",
    AgentType.PORTFOLIO: "portfolio",
    AgentType.MARKET_ANALYSIS: "market_analysis",
    AgentType.GOAL_PLANNING: "goal_planning",
    AgentType.NEWS_SYNTHESIZER: "news_synthesizer",
    AgentType.TAX_EDUCATION: "tax_education",
}


def route_to_agent(state: FinnieState) -> str:
    """Conditional edge: maps next_agent enum → node name."""
    destination = _AGENT_NODE_MAP.get(state.next_agent, "finance_qa")
    logger.info("routing_to_agent", destination=destination)
    return destination


def route_from_classify(state: FinnieState) -> str:
    """Conditional edge out of the merged classifier: refusal or an agent node."""
    if route_after_classify(state) == "rejected":
        return "rejected"
    return route_to_agent(state)


# ── Graph construction ────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def build_graph(persistent: bool = False):
    """Build and compile the LangGraph workflow. Cached per process and variant.

    persistent=False  stateless: each invoke carries its own history. Used by
                      one-shot callers (Portfolio/Market/Goals tabs, evals).
    persistent=True   compiled with the checkpointer: a thread_id in the run
                      config keeps the conversation across turns, refreshes
                      and restarts. Used by the chat page.

    Both variants get the store, so the hydrate node can load the signed-in
    user's profile, holdings and saved plan.

    Shape depends on two independently reversible flags in ``config.yaml``:

      fast_path.faq_cache.enabled  — prepend the semantic answer cache, which
                                     ends the turn on a hit (zero LLM calls) and
                                     appends a write node after every agent.
      fast_path.merged_classifier  — one structured guardrail+router call instead
                                     of the two legacy serial calls.
    """
    settings = get_settings()
    use_cache = settings.fast_path.faq_cache.enabled
    use_merged = settings.fast_path.merged_classifier

    graph = StateGraph(FinnieState)

    # Agent nodes
    graph.add_node("finance_qa", finance_qa_node)
    graph.add_node("portfolio", portfolio_node)
    graph.add_node("market_analysis", market_analysis_node)
    graph.add_node("goal_planning", goal_planning_node)
    graph.add_node("news_synthesizer", news_synthesizer_node)
    graph.add_node("tax_education", tax_education_node)

    # Classification: merged single call, or the legacy guardrail → router pair.
    if use_merged:
        graph.add_node("classify", classify_node)
        entry_node = "classify"
    else:
        graph.add_node("guardrail", guardrail_node)
        graph.add_node("router", router_node)
        entry_node = "guardrail"

    # Load the user's saved data before classification. After the FAQ cache,
    # so a cache hit (which uses no personal context) costs no store read.
    graph.add_node("hydrate", hydrate_node)
    graph.add_edge("hydrate", entry_node)

    # Entry: the FAQ cache short-circuits before any classification.
    if use_cache:
        graph.add_node("faq_cache", faq_cache_node)
        graph.add_node("faq_cache_write", faq_cache_write_node)
        graph.add_edge(START, "faq_cache")
        graph.add_conditional_edges(
            "faq_cache",
            route_after_faq_cache,
            {"hit": END, "miss": "hydrate"},
        )
    else:
        graph.add_edge(START, "hydrate")

    if use_merged:
        # One conditional hop: refusal → END, otherwise straight to the agent.
        graph.add_conditional_edges(
            "classify",
            route_from_classify,
            {"rejected": END, **{name: name for name in AGENT_NODES}},
        )
    else:
        graph.add_conditional_edges(
            "guardrail",
            route_after_guardrail,
            {"allowed": "router", "rejected": END},
        )
        graph.add_conditional_edges(
            "router",
            route_to_agent,
            {name: name for name in AGENT_NODES},
        )

    # Agents → cache write (when enabled) → END
    for node in AGENT_NODES:
        graph.add_edge(node, "faq_cache_write" if use_cache else END)
    if use_cache:
        graph.add_edge("faq_cache_write", END)

    compiled = graph.compile(
        checkpointer=get_checkpointer() if persistent else None,
        store=get_store(),
    )
    logger.info(
        "langgraph_workflow_compiled",
        merged_classifier=use_merged,
        faq_cache=use_cache,
        persistent=persistent,
    )
    return compiled


def _initial_state(
    user_message: str,
    conversation_history: list | None,
    user_profile: dict | None,
) -> FinnieState:
    from src.core.state import FinancialData

    history = list(conversation_history or [])
    history.append(HumanMessage(content=user_message))
    profile = UserProfile(**(user_profile or {})) if user_profile else UserProfile()

    return FinnieState(
        messages=history,
        user_profile=profile,
        financial_data=FinancialData(),
    )


def _check_thread_owner(user_id: Optional[str], thread_id: str) -> None:
    """A thread may only be used by the user whose id prefixes it.

    Thread ids are minted server-side by UserData ("<user_id>:<uuid>"); this
    guard turns a wiring bug that passes the wrong pair into a loud error
    instead of one user reading another's conversation.
    """
    if not user_id or not thread_id.startswith(f"{user_id}:"):
        raise ValueError("thread_id does not belong to user_id")


def _prepare_turn(
    user_message: str,
    conversation_history: list | None,
    user_profile: dict | None,
    user_id: Optional[str],
    thread_id: Optional[str],
) -> tuple[Any, Any, dict[str, Any], list]:
    """Pick the graph variant and build (graph, input, config, fallback_messages)."""
    configurable: dict[str, Any] = {}
    if user_id:
        configurable["user_id"] = user_id

    if thread_id:
        _check_thread_owner(user_id, thread_id)
        configurable["thread_id"] = thread_id
        profile = UserProfile(**user_profile) if user_profile else UserProfile()
        # The checkpoint holds the history, so send only this turn's message —
        # and reset every other field, which would otherwise carry over from
        # the previous turn (see turn_state_reset).
        turn = {
            **turn_state_reset(),
            "user_profile": profile,
            "messages": [HumanMessage(content=user_message)],
        }
        return build_graph(persistent=True), turn, {"configurable": configurable}, turn["messages"]

    state = _initial_state(user_message, conversation_history, user_profile)
    return build_graph(), state, {"configurable": configurable}, state.messages


def _format_result(result: dict[str, Any], fallback_messages: list) -> dict[str, Any]:
    from src.core.state import FinancialData

    next_agent = result.get("next_agent", AgentType.FINANCE_QA)
    financial_data = result.get("financial_data", FinancialData())
    return {
        "final_response": result.get("final_response", ""),
        "agent_used": next_agent.value if hasattr(next_agent, "value") else str(next_agent),
        "router_reasoning": result.get("router_reasoning", ""),
        "financial_data": financial_data.model_dump()
        if hasattr(financial_data, "model_dump")
        else dict(financial_data or {}),
        "rag_context": result.get("rag_context", []),
        "cache_hit": bool(result.get("cache_hit", False)),
        "messages": result.get("messages", fallback_messages),
    }


def _error_result(exc: Exception, fallback_messages: list) -> dict[str, Any]:
    logger.error("workflow_error", error=str(exc))
    return {
        "final_response": f"I encountered an error: {exc}. Please try again.",
        "agent_used": "error",
        "router_reasoning": "",
        "financial_data": {},
        "rag_context": [],
        "cache_hit": False,
        "messages": fallback_messages,
    }


def run_workflow(
    user_message: str,
    conversation_history: list | None = None,
    user_profile: dict | None = None,
    *,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Main entry point for running a single turn through the workflow.

    user_id    the signed-in user (from the authenticated identity, never from
               page input). Their saved profile, holdings and plan replace
               `user_profile` for this turn.
    thread_id  a persistent conversation owned by `user_id`. The history comes
               from the checkpoint, so `conversation_history` is ignored.
               Omit for a one-shot turn.

    Returns a dict with keys: final_response, agent_used, router_reasoning,
    financial_data, rag_context, cache_hit, messages.
    """
    graph, turn, config, fallback = _prepare_turn(
        user_message, conversation_history, user_profile, user_id, thread_id
    )
    try:
        result = graph.invoke(turn, config)
        return _format_result(result, fallback)
    except Exception as exc:  # noqa: BLE001
        return _error_result(exc, fallback)


def stream_workflow(
    user_message: str,
    conversation_history: list | None = None,
    user_profile: dict | None = None,
    sink: Optional[dict[str, Any]] = None,
    *,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Iterator[str]:
    """Yield the answer as it is generated; fill `sink` with the full result.

    Same arguments as run_workflow. Only tokens produced inside an agent node
    are yielded — the classifier's own LLM call is filtered out. A FAQ cache
    hit produces no tokens at all, so the cached answer is emitted in one
    piece. The trailing disclaimer, which agents append after the model call,
    is flushed once the graph finishes.
    """
    graph, turn, config, fallback = _prepare_turn(
        user_message, conversation_history, user_profile, user_id, thread_id
    )
    streamed: list[str] = []
    final_state: dict[str, Any] = {}

    try:
        for mode, payload in graph.stream(
            turn, config=config, stream_mode=["messages", "values"]
        ):
            if mode == "values":
                final_state = payload
                continue

            chunk, metadata = payload
            if metadata.get("langgraph_node") not in AGENT_NODES:
                continue
            # message_text handles both plain-string and content-block chunks.
            # (Reading chunk.text directly calls a deprecated LangChain method.)
            text = message_text(chunk)
            if text:
                streamed.append(text)
                yield text

        result = _format_result(final_state, fallback)
        # Emit whatever the agent added after the model call (the disclaimer),
        # or the whole answer when nothing streamed (cache hit).
        rendered = "".join(streamed)
        response = result["final_response"]
        if response and response != rendered:
            yield response[len(rendered):] if response.startswith(rendered) else response
        if sink is not None:
            sink.update(result)

    except Exception as exc:  # noqa: BLE001
        result = _error_result(exc, fallback)
        if sink is not None:
            sink.update(result)
        yield result["final_response"]


def load_conversation(user_id: str, thread_id: str) -> list[BaseMessage]:
    """The saved messages of a user's thread, for redrawing the chat after a refresh."""
    _check_thread_owner(user_id, thread_id)
    snapshot = build_graph(persistent=True).get_state({"configurable": {"thread_id": thread_id}})
    return list((snapshot.values or {}).get("messages", []))


def delete_conversation(user_id: str, thread_id: str) -> None:
    """Permanently delete a user's saved thread (the chat page's Clear button)."""
    _check_thread_owner(user_id, thread_id)
    build_graph(persistent=True).checkpointer.delete_thread(thread_id)
