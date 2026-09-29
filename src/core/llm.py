"""LLM factory — returns a configured ChatAnthropic instance."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

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


def token_usage(response: Any) -> dict[str, int]:
    """Token accounting for one response, including prompt-cache activity.

    Returns ``input_tokens`` (the uncached remainder, billed at full price),
    ``cache_read_input_tokens`` (~0.1x), ``cache_creation_input_tokens``
    (~1.25x for the 5-minute TTL) and ``output_tokens`` — the same names and
    meaning as Anthropic's ``usage`` object. Total prompt size is the sum of
    the three input fields.

    Prefers the raw Anthropic ``usage`` in ``response_metadata``. Streamed
    responses may carry only LangChain's ``usage_metadata``, whose
    ``input_tokens`` *includes* the cache fields, so the uncached remainder is
    derived by subtraction there. Returns ``{}`` when no usage is available
    (for example a test double), so callers can log unconditionally.
    """
    raw = (getattr(response, "response_metadata", None) or {})
    raw = raw.get("usage") if isinstance(raw, dict) else None
    if isinstance(raw, dict) and "input_tokens" in raw:
        return {
            "input_tokens": int(raw.get("input_tokens") or 0),
            "cache_read_input_tokens": int(raw.get("cache_read_input_tokens") or 0),
            "cache_creation_input_tokens": int(raw.get("cache_creation_input_tokens") or 0),
            "output_tokens": int(raw.get("output_tokens") or 0),
        }

    meta = getattr(response, "usage_metadata", None)
    if not isinstance(meta, dict):
        return {}
    details = meta.get("input_token_details") or {}
    read = int(details.get("cache_read") or 0)
    created = int(details.get("cache_creation") or 0)
    total_input = int(meta.get("input_tokens") or 0)
    return {
        "input_tokens": max(total_input - read - created, 0),
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": created,
        "output_tokens": int(meta.get("output_tokens") or 0),
    }


# Models that removed sampling params — passing `temperature`/`top_p`/`top_k`
# to these returns HTTP 400. Steer these via prompting instead.
_NO_SAMPLING_PARAMS = ("claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5")


@lru_cache(maxsize=2)
def get_llm(streaming: Optional[bool] = None) -> ChatAnthropic:
    """Return the shared ChatAnthropic client.

    ``streaming`` controls whether ``.invoke()`` issues a streaming request
    internally. It must be on for LangGraph's ``stream_mode="messages"`` to emit
    per-token events from agent nodes — the agents still call ``.invoke()`` and
    still get a complete message back. Pass ``False`` for structured-output
    calls (the classifier), which have no use for token events.

    Defaults to the ``fast_path.streaming`` flag.
    """
    settings = get_settings()
    if streaming is None:
        streaming = settings.fast_path.streaming

    kwargs: dict = {
        "model": settings.llm.model,
        "max_tokens": settings.llm.max_tokens,
        "anthropic_api_key": settings.anthropic_api_key,
        "streaming": streaming,
    }
    # Only send temperature to models that still accept it; newer models 400 on it.
    if not settings.llm.model.startswith(_NO_SAMPLING_PARAMS):
        kwargs["temperature"] = settings.llm.temperature
    return ChatAnthropic(**kwargs)
