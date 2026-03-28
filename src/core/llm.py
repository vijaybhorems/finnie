"""LLM factory — returns a configured ChatAnthropic instance."""
from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from src.core.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
    )
