import logging

from multi_llm_chat.history_utils import (
    content_to_text,
    get_provider_name_from_model,
    history_contains_tools,
    normalize_history_turns,
    prepare_request,
)


def test_get_provider_name_from_model():
    assert get_provider_name_from_model("gpt-3.5-turbo") == "chatgpt"
    assert get_provider_name_from_model("GPT-4") == "chatgpt"
    assert get_provider_name_from_model("gemini-pro") == "gemini"
    assert get_provider_name_from_model("GEMINI-1.5-FLASH") == "gemini"


def test_prepare_request_gemini():
    history = [{"role": "user", "content": "hello"}]
    system_prompt = "You are a helpful assistant."

    # Gemini returns tuple (system_prompt, history)
    result = prepare_request(history, system_prompt, "gemini-pro")
    assert result == (system_prompt, history)

    # Empty system prompt
    result = prepare_request(history, "", "gemini-pro")
    assert result == (None, history)


def test_prepare_request_chatgpt():
    history = [{"role": "user", "content": "hello"}]
    system_prompt = "You are a helpful assistant."

    # ChatGPT prepends system prompt to history
    result = prepare_request(history, system_prompt, "gpt-3.5-turbo")
    assert result == [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "hello"},
    ]

    # Empty system prompt
    result = prepare_request(history, "", "gpt-3.5-turbo")
    assert result == history


def test_content_to_text_includes_tool_data_when_enabled():
    content = [
        {"type": "text", "content": "Hello"},
        {"type": "tool_call", "content": {"name": "search", "arguments": {"q": "tokyo"}}},
        {"type": "tool_result", "name": "search", "content": {"result": "ok"}},
    ]

    assert content_to_text(content, include_tool_data=False) == "Hello"
    with_tools = content_to_text(content, include_tool_data=True)
    assert "Hello" in with_tools
    assert "search" in with_tools
    assert "tokyo" in with_tools


def test_normalize_history_turns_logs_on_invalid_entry(caplog):
    """Invalid entries should be replaced with placeholders to preserve index alignment."""
    caplog.set_level(logging.WARNING)
    history = [{"role": "user", "content": "hello"}, "invalid", {"role": "gemini", "content": "hi"}]

    normalized = normalize_history_turns(history)

    # All entries should be preserved (including placeholder)
    assert len(normalized) == 3
    assert normalized[0]["role"] == "user"
    assert normalized[1]["role"] == "system"  # Placeholder
    assert normalized[1]["invalid"] is True
    assert normalized[1]["content"] == [{"type": "text", "content": "[Invalid entry removed]"}]
    assert normalized[2]["role"] == "gemini"
    assert "Invalid history entry at index 1" in caplog.text


def test_normalize_history_turns_preserves_tool_name():
    history = [{"role": "tool", "name": "search", "content": "ok"}]

    normalized = normalize_history_turns(history)

    assert normalized[0]["content"] == [{"type": "tool_result", "name": "search", "content": "ok"}]


def test_normalize_history_turns_preserves_index_alignment():
    """Placeholder replacement should preserve index alignment for get_llm_response()."""
    from multi_llm_chat.history import get_llm_response

    # History with invalid entry at index 1
    history = [
        {"role": "user", "content": "first"},
        "invalid_entry",  # This should become a placeholder
        {"role": "gemini", "content": "response"},
    ]

    normalized = normalize_history_turns(history)

    # Index should be preserved
    assert len(normalized) == 3
    assert get_llm_response(normalized, 0) == "response"  # Still at the correct index

    # Placeholder should be identifiable
    assert normalized[1].get("invalid") is True


def test_content_to_text_japanese_characters_not_escaped():
    """Japanese characters should not be escaped as \\uXXXX (Issue #79 Review Fix)."""
    content = [
        {"type": "text", "content": "こんにちは"},
        {
            "type": "tool_call",
            "content": {"name": "search", "arguments": {"query": "東京天気"}},
        },
    ]

    result = content_to_text(content, include_tool_data=True)

    # Japanese should NOT be escaped
    assert "こんにちは" in result
    assert "東京天気" in result
    # Should NOT have Unicode escapes like \u3053
    assert "\\u" not in result


def test_history_contains_tools_with_tool_calls():
    """Tool calls should be detected in history"""
    history = [
        {"role": "user", "content": "Search for Tokyo weather"},
        {
            "role": "gemini",
            "content": [
                {"type": "tool_call", "content": {"name": "search", "arguments": {"q": "tokyo"}}}
            ],
        },
    ]
    assert history_contains_tools(history) is True


def test_history_contains_tools_with_tool_results():
    """Tool results should be detected in history"""
    history = [
        {"role": "user", "content": "Search for Tokyo weather"},
        {
            "role": "tool",
            "content": [{"type": "tool_result", "content": {"result": "Sunny"}}],
        },
    ]
    assert history_contains_tools(history) is True


def test_history_contains_tools_without_tools():
    """Regular conversation should return False"""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "gemini", "content": "Hi there!"},
    ]
    assert history_contains_tools(history) is False


def test_history_contains_tools_empty_history():
    """Empty history should return False"""
    assert history_contains_tools([]) is False
