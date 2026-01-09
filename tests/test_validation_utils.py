import pytest
import os
from unittest.mock import patch
from multi_llm_chat.validation import validate_system_prompt_length, validate_context_length

def test_validate_system_prompt_length():
    # Normal case
    result = validate_system_prompt_length("Helpful assistant", "gemini-pro")
    assert result["valid"] is True
    
    # Exceeds limit
    long_prompt = "A" * 50000
    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "1000"}):
        result = validate_system_prompt_length(long_prompt, "gemini-pro")
        assert result["valid"] is False
        assert "exceeds" in result["error"].lower()

def test_validate_context_length():
    system_prompt = "You are a helpful assistant."
    history = [
        {"role": "user", "content": "A" * 5000},
    ]

    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "1000"}):
        result = validate_context_length(history, system_prompt, model_name="gemini-pro")
        assert result["valid"] is False
        assert "too long" in result["error"].lower()
