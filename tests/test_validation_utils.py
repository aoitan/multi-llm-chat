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


def test_validate_context_length_structured_content():
    system_prompt = "You are a helpful assistant."
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "content": "Hello"},
                {"type": "text", "content": "world"},
            ],
        },
        {
            "role": "gemini",
            "content": [{"type": "text", "content": "Response"}],
        },
    ]

    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "1000"}):
        result = validate_context_length(
            history, system_prompt, model_name="gemini-pro", token_calculator=mock_calculate_tokens
        )
        assert result["valid"] is True


def test_validate_context_length_with_large_tool_result():
    """大きなツール結果がコンテキスト長を超過する場合の検証

    Note: validate_context_length() only validates the latest turn,
    so we place the large tool result in the latest exchange.
    """
    system_prompt = "You are a helpful assistant."

    # Simulate tool result with 5000 tokens worth of data in latest turn
    large_result = "x" * 20000  # Approximate 5000 tokens (4 chars per token)
    history = [
        {"role": "user", "content": [{"type": "text", "content": "First query"}]},
        {
            "role": "gemini",
            "content": [{"type": "text", "content": "First response"}],
        },
        # Latest turn starts here
        {"role": "user", "content": [{"type": "text", "content": "Search"}]},
        {
            "role": "gemini",
            "content": [
                {
                    "type": "text",
                    "content": large_result,  # Gemini's response with large data
                }
            ],
        },
    ]

    # Set max context to 3000 tokens (should fail validation due to large response)
    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "3000"}):
        result = validate_context_length(
            history,
            system_prompt,
            model_name="gemini-pro",
            token_calculator=mock_calculate_tokens,
        )
        assert result["valid"] is False
        assert "too long" in result["error"].lower()


def test_validate_context_length_with_tool_call_and_result():
    """ツール呼び出しと結果を含む履歴の検証（正常系）

    Note: This test verifies that content_to_text() correctly extracts
    text from tool calls for token calculation.
    """
    system_prompt = "You are a helpful assistant."
    history = [
        {"role": "user", "content": [{"type": "text", "content": "Search for Python"}]},
        {
            "role": "gemini",
            "content": [
                {
                    "type": "tool_call",
                    "content": {"name": "search", "arguments": {"query": "python"}},
                },
                {"type": "text", "content": "Python is a programming language."},
            ],
        },
    ]

    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "10000"}):
        result = validate_context_length(
            history,
            system_prompt,
            model_name="gemini-pro",
            token_calculator=mock_calculate_tokens,
        )
        assert result["valid"] is True
