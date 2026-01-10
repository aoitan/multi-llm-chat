import os
from unittest.mock import patch

from multi_llm_chat.validation import validate_context_length, validate_system_prompt_length


def mock_calculate_tokens(text: str, model_name: str) -> int:
    return len(text)


def test_validate_system_prompt_length():
    # Normal case
    result = validate_system_prompt_length(
        "Helpful assistant", "gemini-pro", token_calculator=mock_calculate_tokens
    )
    assert result["valid"] is True

    # Exceeds limit
    # Mock calculation: 1 char = 1 token
    long_prompt = "A" * 1500
    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "1000"}):
        result = validate_system_prompt_length(
            long_prompt, "gemini-pro", token_calculator=mock_calculate_tokens
        )
        assert result["valid"] is False
        assert "exceeds" in result["error"].lower()


def test_validate_context_length():
    system_prompt = "You are a helpful assistant."
    history = [
        {"role": "user", "content": "A" * 5000},
    ]

    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "1000"}):
        result = validate_context_length(
            history, system_prompt, model_name="gemini-pro", token_calculator=mock_calculate_tokens
        )
        assert result["valid"] is False
        assert "too long" in result["error"].lower()
