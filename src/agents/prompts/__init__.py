"""Agent system prompts.

Prompts live in this directory as Markdown so they can be reviewed, diffed and
counted in isolation from the code that assembles them.

  finnie_core.md   Shared by all six agents: persona, guidelines, how to use the
                   request context. Opens every system prompt.
  <agent>.md       One per agent: that agent's task instructions.

The system prompt an agent sends is three blocks, most stable first, because
Anthropic prompt caching is a prefix match:

  1. shared_system_prompt()  finnie_core.md + knowledge-base digest  [cached]
  2. the agent's role block  name, description, <agent>.md, static   [cached]
                             reference data
  3. request context         user profile, retrieved passages, live  [not cached]
                             data — different on every call

Block 1 is byte-identical across agents, so one cache entry serves them all.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.rag.digest import knowledge_base_digest

_PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the text of ``<name>.md`` from this directory."""
    return (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def shared_system_prompt() -> str:
    """The block every agent's system prompt opens with — identical bytes for all."""
    return f"{load_prompt('finnie_core')}\n\n{knowledge_base_digest()}"
