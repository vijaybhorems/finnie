"""Prompt caching: system prompt structure, config, usage accounting, digest.

A broken cache fails silently — requests still succeed, they just cost more —
so these tests pin the invariants that keep the cached prefix reusable:

  * the two stable blocks carry cache_control and the per-request block does not;
  * the shared block is byte-identical across all six agents;
  * the stable blocks are byte-identical across requests with different data;
  * nothing request-specific (question, profile, retrieved text, live data)
    ever lands in a stable block.

Whether the API actually reads the cache is verified live in
tests/evals/test_prompt_cache_evals.py.
"""
from __future__ import annotations

import itertools
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.llm import token_usage
from src.core.state import FinancialData, FinnieState, UserProfile
from src.rag.digest import _is_section_heading, _summarize, knowledge_base_digest
from tests.test_agent_runs import _AGENTS, _IDS, RecordingChatModel, run_agent

_EPHEMERAL = {"type": "ephemeral"}


def _system_blocks(llm: RecordingChatModel) -> list[dict]:
    return llm.calls[0][0].content


# ── Structure of what each agent sends ─────────────────────────────────────

@pytest.mark.parametrize("name,module,cls,patcher,query", _AGENTS, ids=_IDS)
class TestSystemPromptStructure:
    def test_stable_blocks_cached_context_block_not(self, name, module, cls, patcher, query):
        _, llm = run_agent(module, cls, patcher, query)
        shared, role, context = _system_blocks(llm)

        assert shared["cache_control"] == _EPHEMERAL
        assert role["cache_control"] == _EPHEMERAL
        assert "cache_control" not in context
        assert context["text"].startswith("# Context for this request")

    def test_request_data_only_in_context_block(self, name, module, cls, patcher, query):
        _, llm = run_agent(module, cls, patcher, query)
        shared, role, context = _system_blocks(llm)
        stable = shared["text"] + role["text"]

        # The profile line is request data; so is anything fetched per call.
        assert "User profile:" in context["text"]
        assert "User profile:" not in stable
        assert query not in stable

    def test_task_instructions_in_role_block(self, name, module, cls, patcher, query):
        from src.agents.prompts import load_prompt

        _, llm = run_agent(module, cls, patcher, query)
        _, role, context = _system_blocks(llm)
        instructions = load_prompt(name)

        assert instructions in role["text"]
        assert instructions not in context["text"]


def test_shared_block_identical_across_all_agents():
    """One cache entry can only serve every agent if the bytes match exactly."""
    shared_texts = {
        _system_blocks(run_agent(module, cls, patcher, query)[1])[0]["text"]
        for _, module, cls, patcher, query in _AGENTS
    }
    assert len(shared_texts) == 1


def test_role_blocks_are_distinct_per_agent():
    role_texts = {
        _system_blocks(run_agent(module, cls, patcher, query)[1])[1]["text"]
        for _, module, cls, patcher, query in _AGENTS
    }
    assert len(role_texts) == len(_AGENTS)


def test_stable_blocks_identical_across_requests():
    """Different questions and data must not perturb the cached prefix."""
    _, module, cls, patcher, _ = _AGENTS[0]  # finance_qa: RAG + macro, the most data

    def _different_data(stack):
        retriever = stack.enter_context(patch("src.agents.finance_qa_agent.get_retriever"))
        retriever.return_value.get_context.return_value = (
            "[Source 1: Bond Basics (investing_basics/bonds.txt)]\nPrices fall as rates rise."
        )
        fred = stack.enter_context(patch("src.agents.finance_qa_agent.FredClient"))
        fred.return_value.get_macro_snapshot.return_value = {
            "fed_funds_rate": {"value": 5.25, "date": "2026-03-01"},
        }

    _, first = run_agent(module, cls, patcher, "What is a P/E ratio?")
    _, second = run_agent(module, cls, _different_data, "How do bonds work when rates rise?")

    a, b = _system_blocks(first), _system_blocks(second)
    assert a[0] == b[0]
    assert a[1] == b[1]
    # The volatile block did move — so the stable blocks holding still is meaningful.
    assert a[2] != b[2]


def test_tax_reference_data_is_in_cached_role_block():
    _, module, cls, patcher, query = next(a for a in _AGENTS if a[0] == "tax_education")
    _, llm = run_agent(module, cls, patcher, query)
    _, role, context = _system_blocks(llm)

    assert "2024 TAX REFERENCE DATA" in role["text"]
    assert "2024 TAX REFERENCE DATA" not in context["text"]


# ── Config ─────────────────────────────────────────────────────────────────

def _finance_agent():
    with patch("src.agents.finance_qa_agent.get_retriever"), \
         patch("src.agents.finance_qa_agent.FredClient"), \
         patch("src.agents.base_agent.get_llm"):
        from src.agents.finance_qa_agent import FinanceQAAgent
        return FinanceQAAgent()


def _state(query: str = "hi") -> FinnieState:
    return FinnieState(
        messages=[HumanMessage(content=query)],
        user_profile=UserProfile(),
        financial_data=FinancialData(),
    )


class TestCachingConfig:
    def test_disabled_sends_no_cache_markers(self, monkeypatch):
        from src.core.config import get_settings

        monkeypatch.setattr(get_settings().llm.prompt_caching, "enabled", False)
        blocks = _finance_agent()._build_messages(_state(), "some context")[0].content

        assert all("cache_control" not in block for block in blocks)

    def test_one_hour_ttl(self, monkeypatch):
        from src.core.config import get_settings

        monkeypatch.setattr(get_settings().llm.prompt_caching, "ttl", "1h")
        shared, role, _ = _finance_agent()._build_messages(_state(), "ctx")[0].content

        # Both markers share one TTL, so the longer-before-shorter rule holds.
        assert shared["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert role["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_empty_context_omits_the_block(self):
        """The API rejects empty text blocks."""
        blocks = _finance_agent()._build_messages(_state(), "   \n ")[0].content
        assert len(blocks) == 2

    def test_config_yaml_defaults(self):
        from src.core.config import get_settings

        caching = get_settings().llm.prompt_caching
        assert caching.enabled is True
        assert caching.ttl == "5m"


# ── Usage accounting ───────────────────────────────────────────────────────

class TestTokenUsage:
    def test_prefers_raw_anthropic_usage(self):
        response = AIMessage(
            content="x",
            response_metadata={"usage": {
                "input_tokens": 120,
                "cache_read_input_tokens": 1500,
                "cache_creation_input_tokens": 0,
                "output_tokens": 300,
            }},
        )
        assert token_usage(response) == {
            "input_tokens": 120,
            "cache_read_input_tokens": 1500,
            "cache_creation_input_tokens": 0,
            "output_tokens": 300,
        }

    def test_usage_metadata_input_includes_cache_fields(self):
        """LangChain's aggregate input_tokens includes cache reads and writes."""
        response = AIMessage(
            content="x",
            usage_metadata={
                "input_tokens": 1620,
                "output_tokens": 300,
                "total_tokens": 1920,
                "input_token_details": {"cache_read": 1500, "cache_creation": 0},
            },
        )
        usage = token_usage(response)
        assert usage["input_tokens"] == 120
        assert usage["cache_read_input_tokens"] == 1500

    def test_no_usage_returns_empty(self):
        assert token_usage(MagicMock(spec=[])) == {}
        assert token_usage(AIMessage(content="x")) == {}


class TestCacheInactiveWarning:
    @pytest.fixture(autouse=True)
    def _fresh_warned_set(self, monkeypatch):
        monkeypatch.setattr("src.agents.base_agent._CACHE_INACTIVE_WARNED", set())

    def _agent_with_usage(self, cache_read: int, cache_creation: int):
        reply = AIMessage(
            content="answer",
            usage_metadata={
                "input_tokens": 900 + cache_read + cache_creation,
                "output_tokens": 10,
                "total_tokens": 910 + cache_read + cache_creation,
                "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
            },
        )
        llm = RecordingChatModel(messages=itertools.cycle([reply]))
        with patch("src.agents.finance_qa_agent.get_retriever"), \
             patch("src.agents.finance_qa_agent.FredClient"), \
             patch("src.agents.base_agent.get_llm", return_value=llm):
            from src.agents.finance_qa_agent import FinanceQAAgent
            agent = FinanceQAAgent()
        agent._logger = MagicMock()
        return agent

    def _warnings(self, agent) -> list:
        return [c for c in agent._logger.warning.call_args_list if c.args[0] == "prompt_cache_inactive"]

    def test_warns_once_when_nothing_cached(self):
        agent = self._agent_with_usage(cache_read=0, cache_creation=0)
        agent._invoke_llm(_state(), "ctx")
        agent._invoke_llm(_state(), "ctx")

        assert len(self._warnings(agent)) == 1

    @pytest.mark.parametrize("read,created", [(1500, 0), (0, 1500)])
    def test_silent_when_cache_read_or_written(self, read, created):
        agent = self._agent_with_usage(cache_read=read, cache_creation=created)
        agent._invoke_llm(_state(), "ctx")

        assert self._warnings(agent) == []

    def test_usage_fields_logged_on_success(self):
        agent = self._agent_with_usage(cache_read=1500, cache_creation=0)
        agent._invoke_llm(_state(), "ctx")

        logged = agent._logger.info.call_args
        assert logged.args[0] == "llm_call_success"
        assert logged.kwargs["cache_read_input_tokens"] == 1500
        assert logged.kwargs["input_tokens"] == 900


# ── Knowledge-base digest ──────────────────────────────────────────────────

class TestDigest:
    @pytest.mark.parametrize("line,expected", [
        ("THE POWER OF STARTING EARLY", True),
        ("SHORT-TERM vs. LONG-TERM CAPITAL GAINS", True),
        ("LEADING INDICATORS (Predict future economic activity)", True),
        ("401(k) PLANS", True),
        ("What Is an Index Fund?", False),
        ("SPY, VOO, IVV: Track the S&P 500", False),
        ("Examples", False),
    ])
    def test_section_heading_detection(self, line, expected):
        assert _is_section_heading(line) is expected

    def test_summary_is_first_paragraph_and_truncated(self):
        long_paragraph = " ".join(f"word{i}" for i in range(100))
        title, summary, sections = _summarize(
            f"My Article\n\n{long_paragraph}\n\nFIRST SECTION\nbody\n\nSECOND SECTION\nbody"
        )
        assert title == "My Article"
        assert summary.startswith("word0 word1") and summary.endswith(" …")
        assert len(summary.split()) == 61  # 60 words + ellipsis
        assert sections == ["FIRST SECTION", "SECOND SECTION"]

    def test_covers_every_article(self):
        from src.core.config import get_settings

        digest = knowledge_base_digest()
        kb = get_settings().knowledge_base_path
        for article in kb.rglob("*.txt"):
            title = article.read_text(encoding="utf-8").splitlines()[0].strip()
            assert f"**{title}**" in digest

    def test_deterministic(self):
        first = knowledge_base_digest()
        knowledge_base_digest.cache_clear()
        assert knowledge_base_digest() == first
