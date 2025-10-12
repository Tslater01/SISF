# tests/test_adjudicator.py
"""
Unit tests for the Adjudicator.
"""
import pytest
from unittest.mock import patch, MagicMock
from sisf.components.adjudicator import EnsembleAdjudicator, FailureCategory
import json

@pytest.fixture
def adjudicator():
    return EnsembleAdjudicator(api_key="FAKE_KEY")

@patch("openai.resources.chat.completions.Completions.create")
def test_adjudicator_detects_breach(mock_create, adjudicator):
    mock_api_response = {
        "is_breach": True,
        "reasoning": "The model provided harmful instructions.",
        "failure_category": "HarmfulContent"
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_api_response)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_create.return_value = mock_completion
    
    result = adjudicator.analyze("bad prompt", "bad response")
    
    assert result.is_breach is True
    assert result.failure_category == FailureCategory.HARMFUL_CONTENT

@patch("openai.resources.chat.completions.Completions.create")
def test_adjudicator_clears_safe_prompt(mock_create, adjudicator):
    mock_api_response = {
        "is_breach": False,
        "reasoning": "The model correctly refused the harmful request.",
        "failure_category": "None"
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_api_response)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_create.return_value = mock_completion
    
    result = adjudicator.analyze("bad prompt", "good response")
    
    assert result.is_breach is False
    assert result.failure_category == FailureCategory.NONE

@patch("openai.resources.chat.completions.Completions.create")
def test_adjudicator_handles_api_error(mock_create, adjudicator):
    """Tests that the adjudicator returns its safe, error-state default on API failure."""
    mock_create.side_effect = Exception("API connection timed out")
    
    result = adjudicator.analyze("any prompt", "any response")
    
    assert result.is_breach is False
    assert result.failure_category == FailureCategory.ERROR
    assert "timed out" in result.reasoning