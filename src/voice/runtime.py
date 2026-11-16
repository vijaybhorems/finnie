"""The settings a voice process runs with, where they differ from chat.

Pure (no LiveKit): the worker calls `configure_process()` at import, and the
voice evals call it too, so they measure the models the worker really uses.
"""
from __future__ import annotations

import os

from src.core.config import get_settings


def configure_process() -> None:
    """Apply the voice settings that differ from chat to this process only.

    Must run before the first get_llm() or get_pool() call, which cache what
    they read. FINNIE_LLM_THINKING / FINNIE_CLASSIFIER_MODEL still win when set,
    so the chat-page behaviour can be measured from the worker.
    """
    settings = get_settings()
    voice = settings.voice
    # Voice can't afford the model thinking silently before it speaks (~9 s on goals questions).
    if not os.environ.get("FINNIE_LLM_THINKING"):
        settings.llm.thinking = voice.llm_thinking
    # Same routing and scope accuracy as Sonnet 5 on the labelled cases, ~0.45 s
    # faster per turn (spike/voice/compare_classifier.py).
    if not os.environ.get("FINNIE_CLASSIFIER_MODEL"):
        settings.llm.classifier_model = voice.classifier_model
    settings.persistence.pool_max_size = min(settings.persistence.pool_max_size, voice.db_pool_max_size)
