"""llm.thinking: the default sends nothing; "disabled" turns thinking off (voice worker)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import get_settings
from src.core.llm import get_llm


def test_default_sends_no_thinking_parameter():
    assert get_settings().llm.thinking == "adaptive"
    assert not get_llm().thinking


def test_env_override_disables_thinking(monkeypatch):
    monkeypatch.setenv("FINNIE_LLM_THINKING", "disabled")
    assert get_settings().llm.thinking == "disabled"
    assert get_llm().thinking == {"type": "disabled"}
    assert get_llm(streaming=False).thinking == {"type": "disabled"}  # the classifier too


def test_unknown_value_is_rejected(monkeypatch):
    monkeypatch.setenv("FINNIE_LLM_THINKING", "off")
    with pytest.raises(ValidationError):
        get_settings()


def test_classifier_uses_its_own_model_when_set(monkeypatch):
    from unittest.mock import patch

    from langchain_core.messages import HumanMessage

    from src.core.state import FinancialData, FinnieState, UserProfile
    from src.workflow.classify import classify_node

    monkeypatch.setenv("FINNIE_CLASSIFIER_MODEL", "claude-haiku-5-5")
    assert get_settings().llm.classifier_model == "claude-haiku-5-5"
    with patch("src.workflow.classify.get_llm") as mock_get_llm:
        classify_node(FinnieState(messages=[HumanMessage(content="What is a P/E ratio?")],
                                  user_profile=UserProfile(), financial_data=FinancialData()))
    mock_get_llm.assert_called_once_with(streaming=False, model="claude-haiku-5-5")


def test_haiku_gets_no_sampling_params():
    llm = get_llm(streaming=False, model="claude-haiku-5-5")
    assert llm.model == "claude-haiku-5-5"
    assert llm.temperature is None
