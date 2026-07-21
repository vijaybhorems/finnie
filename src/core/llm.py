"""LLM factory — returns a configured ChatAnthropic instance."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic

from src.core.config import get_settings


def message_text(response: Any) -> str:
    """Extract plain text from an LLM response.

    Anthropic models can return ``.content`` either as a plain string or as a
    list of content blocks (e.g. ``[{"type": "text", "text": "..."}]``, plus
    thinking/tool blocks). Callers that do ``content + "..."`` or
    ``content.strip()`` break on the list form, so normalise here: join the text
    of every text block and drop non-text blocks.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Only text blocks; skip thinking/tool_use/etc.
                if block.get("type", "text") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:  # object-style block with a .text attribute
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


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
