"""LLM factory — returns a configured ChatAnthropic instance."""
from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from src.core.config import get_settings


# Models that removed sampling params — passing `temperature`/`top_p`/`top_k`
# to these returns HTTP 400. Steer these via prompting instead.
_NO_SAMPLING_PARAMS = ("claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5")


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    settings = get_settings()
    kwargs: dict = {
        "model": settings.llm.model,
        "max_tokens": settings.llm.max_tokens,
        "anthropic_api_key": settings.anthropic_api_key,
    }
    # Only send temperature to models that still accept it; newer models 400 on it.
    if not settings.llm.model.startswith(_NO_SAMPLING_PARAMS):
        kwargs["temperature"] = settings.llm.temperature
    return ChatAnthropic(**kwargs)
