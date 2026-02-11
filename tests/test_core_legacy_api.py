"""Tests for core_modules/legacy_api.py - Legacy API functions

⚠️ LEGACY API TESTS - These functions are deprecated and will be removed in v2.0.0

This module tests the legacy (deprecated) API functions,
including provider-specific formatting, API calls, streaming, and text extraction.
These functions maintain backward compatibility with older code.
"""

from unittest.mock import Mock, patch

import pytest

import multi_llm_chat.core as core


# Issue #115: Test that DeprecationWarnings are raised
@pytest.mark.legacy
def test_legacy_functions_raise_deprecation_warnings():
    """All legacy API functions should raise DeprecationWarning when called"""
    history = [{"role": "user", "content": "test"}]

    with pytest.warns(DeprecationWarning, match="prepare_request.*deprecated"):
        core.prepare_request(history, None, "gemini-2.0-flash-exp")

    with pytest.warns(DeprecationWarning, match="format_history_for_gemini.*deprecated"):
        core.format_history_for_gemini(history)

    with pytest.warns(DeprecationWarning, match="format_history_for_chatgpt.*deprecated"):
        core.format_history_for_chatgpt(history)

    with pytest.warns(DeprecationWarning, match="extract_text_from_chunk.*deprecated"):
        core.extract_text_from_chunk("test", "gemini-2.0-flash-exp")

    with pytest.warns(DeprecationWarning, match="load_api_key.*deprecated"):
        with patch.dict("os.environ", {"TEST_KEY": "test_value"}):
            core.load_api_key("TEST_KEY")


@pytest.mark.legacy
def test_prepare_request_openai_adds_system_prompt():
    """prepare_request should add system prompt to OpenAI history at the beginning"""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    system_prompt = "You are a helpful assistant."

    result = core.prepare_request(history, system_prompt, "gpt-4o")

    assert len(result) == 3
    assert result[0]["role"] == "system"
    assert result[0]["content"] == system_prompt
    assert result[1] == history[0]
    assert result[2] == history[1]


@pytest.mark.legacy
def test_prepare_request_gemini_returns_tuple():
    """prepare_request should return (system_prompt, history) tuple for Gemini"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    result = core.prepare_request(history, system_prompt, "gemini-2.0-flash-exp")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == system_prompt
    assert result[1] == history


@pytest.mark.legacy
def test_prepare_request_empty_system_prompt_openai():
    """prepare_request should not add system message when system_prompt is empty for OpenAI"""
    history = [{"role": "user", "content": "Hello"}]

    result = core.prepare_request(history, "", "gpt-4o")

    assert result == history


@pytest.mark.legacy
def test_prepare_request_empty_system_prompt_gemini():
    """prepare_request should return (None, history) when system_prompt is empty for Gemini"""
    history = [{"role": "user", "content": "Hello"}]

    result = core.prepare_request(history, "", "gemini-2.0-flash-exp")

    assert isinstance(result, tuple)
    assert result[0] is None
    assert result[1] == history


@pytest.mark.legacy
def test_prepare_request_whitespace_only_system_prompt_openai():
    """prepare_request should not add system message for whitespace-only prompt (OpenAI)"""
    history = [{"role": "user", "content": "Hello"}]

    result = core.prepare_request(history, "   ", "gpt-4o")

    assert result == history


@pytest.mark.legacy
def test_prepare_request_whitespace_only_system_prompt_gemini():
    """prepare_request should return (None, history) for whitespace-only prompt (Gemini)"""
    history = [{"role": "user", "content": "Hello"}]

    result = core.prepare_request(history, "  \t\n  ", "gemini-2.0-flash-exp")

    assert isinstance(result, tuple)
    assert result[0] is None
    assert result[1] == history


@pytest.mark.legacy
def test_load_api_key_reads_from_env():
    """load_api_key should read API key from environment"""
    with patch("os.getenv", return_value="test_key_12345"):
        key = core.load_api_key("GOOGLE_API_KEY")
        assert key == "test_key_12345"


@pytest.mark.legacy
def test_format_history_for_gemini():
    """format_history should convert to Gemini format"""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "gemini", "content": "Hi there"},
    ]

    result = core.format_history_for_gemini(history)

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["parts"] == [{"text": "Hello"}]
    assert result[1]["role"] == "model"
    assert result[1]["parts"] == [{"text": "Hi there"}]


@pytest.mark.legacy
def test_format_history_for_gemini_filters_chatgpt_responses():
    """format_history_for_gemini should filter out ChatGPT responses to avoid confusion"""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "chatgpt", "content": "Hi from ChatGPT"},
        {"role": "user", "content": "Another question"},
        {"role": "gemini", "content": "Answer from Gemini"},
    ]

    result = core.format_history_for_gemini(history)

    # Should only include user messages and Gemini's own responses
    assert len(result) == 3
    assert result[0] == {"role": "user", "parts": [{"text": "Hello"}]}
    assert result[1] == {"role": "user", "parts": [{"text": "Another question"}]}
    assert result[2] == {"role": "model", "parts": [{"text": "Answer from Gemini"}]}


@pytest.mark.legacy
def test_format_history_for_chatgpt():
    """format_history should convert to ChatGPT format"""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "chatgpt", "content": "Hi there"},
    ]

    result = core.format_history_for_chatgpt(history)

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "Hi there"


@pytest.mark.legacy
def test_format_history_for_chatgpt_preserves_system_role():
    """format_history_for_chatgpt should preserve the 'system' role."""
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    result = core.format_history_for_chatgpt(history)

    assert len(result) == 2
    assert result[0]["role"] == "system", "The 'system' role should be preserved"
    assert result[0]["content"] == "You are a helpful assistant."
    assert result[1]["role"] == "user"


@pytest.mark.legacy
def test_format_history_for_chatgpt_filters_gemini_responses():
    """format_history_for_chatgpt should filter out Gemini responses to avoid confusion"""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "gemini", "content": "Hi from Gemini"},
        {"role": "user", "content": "Another question"},
        {"role": "chatgpt", "content": "Answer from ChatGPT"},
    ]

    result = core.format_history_for_chatgpt(history)

    # Should only include user messages and ChatGPT's own responses
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "Hello"}
    assert result[1] == {"role": "user", "content": "Another question"}
    assert result[2] == {"role": "assistant", "content": "Answer from ChatGPT"}


@pytest.mark.legacy
def test_call_gemini_api_with_system_prompt():
    """call_gemini_api should delegate to GeminiProvider with system prompt"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core_modules.legacy_api.create_provider") as mock_create_provider:
        mock_provider = Mock()

        async def mock_call_api(*args, **kwargs):
            yield Mock(text="Response")

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        list(core.call_gemini_api(history, system_prompt))

        mock_create_provider.assert_called_once_with("gemini")
        mock_provider.call_api.assert_called_once_with(history, system_prompt)


@pytest.mark.legacy
def test_call_gemini_api_without_system_prompt():
    """call_gemini_api should delegate to GeminiProvider without system prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core_modules.legacy_api.create_provider") as mock_create_provider:
        mock_provider = Mock()

        async def mock_call_api(*args, **kwargs):
            yield Mock(text="Response")

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        list(core.call_gemini_api(history))

        mock_create_provider.assert_called_once_with("gemini")
        mock_provider.call_api.assert_called_once_with(history, None)


@pytest.mark.legacy
def test_call_chatgpt_api_with_system_prompt():
    """call_chatgpt_api should delegate to ChatGPTProvider with system prompt"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core_modules.legacy_api.create_provider") as mock_create_provider:
        mock_provider = Mock()

        async def mock_call_api(*args, **kwargs):
            yield Mock(text="Response")

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        list(core.call_chatgpt_api(history, system_prompt))

        mock_create_provider.assert_called_once_with("chatgpt")
        mock_provider.call_api.assert_called_once_with(history, system_prompt)


@pytest.mark.legacy
def test_call_chatgpt_api_without_system_prompt():
    """call_chatgpt_api should delegate to ChatGPTProvider without system prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core_modules.legacy_api.create_provider") as mock_create_provider:
        mock_provider = Mock()

        async def mock_call_api(*args, **kwargs):
            yield Mock(text="Response")

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        list(core.call_chatgpt_api(history))

        mock_create_provider.assert_called_once_with("chatgpt")
        mock_provider.call_api.assert_called_once_with(history, None)


@pytest.mark.legacy
def test_stream_text_events_with_system_prompt():
    """stream_text_events should delegate to provider with system prompt"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core_modules.legacy_api.create_provider") as mock_create_provider:
        mock_provider = Mock()

        async def mock_stream_text_events(*args, **kwargs):
            yield "Response"

        mock_provider.stream_text_events.side_effect = mock_stream_text_events
        mock_create_provider.return_value = mock_provider

        result = list(core.stream_text_events(history, "gemini", system_prompt))

        assert result == ["Response"]
        mock_create_provider.assert_called_once_with("gemini")
        mock_provider.stream_text_events.assert_called_once_with(history, system_prompt)


@pytest.mark.legacy
def test_stream_text_events_without_system_prompt():
    """stream_text_events should delegate to provider without system prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core_modules.legacy_api.create_provider") as mock_create_provider:
        mock_provider = Mock()

        async def mock_stream_text_events(*args, **kwargs):
            yield "Response"

        mock_provider.stream_text_events.side_effect = mock_stream_text_events
        mock_create_provider.return_value = mock_provider

        result = list(core.stream_text_events(history, "chatgpt"))

        assert result == ["Response"]
        mock_create_provider.assert_called_once_with("chatgpt")
        mock_provider.stream_text_events.assert_called_once_with(history, None)


@pytest.mark.legacy
def test_extract_text_from_chunk_gemini():
    """extract_text_from_chunk should extract text from Gemini chunk"""
    # Create a new dictionary-based chunk
    chunk = {"type": "text", "content": "Hello from Gemini"}
    # The model name 'gemini' will cause the GeminiProvider to be used
    result = core.extract_text_from_chunk(chunk, "gemini")
    assert result == "Hello from Gemini"

    # Test with a non-text chunk, which should return empty
    chunk = {"type": "tool_call", "content": {}}
    result = core.extract_text_from_chunk(chunk, "gemini")
    assert result == ""


@pytest.mark.legacy
def test_extract_text_from_chunk_chatgpt_string():
    """extract_text_from_chunk should extract text from ChatGPT string content"""
    # Create a mock ChatGPT chunk with string content
    delta = type("Delta", (), {"content": "Hello from ChatGPT"})()
    choice = type("Choice", (), {"delta": delta})()
    chunk = type("Chunk", (), {"choices": [choice]})()

    result = core.extract_text_from_chunk(chunk, "chatgpt")
    assert result == "Hello from ChatGPT"


@pytest.mark.legacy
def test_extract_text_from_chunk_chatgpt_list():
    """extract_text_from_chunk should extract text from ChatGPT list content"""
    # Create a mock ChatGPT chunk with list content
    part1 = type("Part", (), {"text": "Hello "})()
    part2 = type("Part", (), {"text": "from ChatGPT"})()
    delta = type("Delta", (), {"content": [part1, part2]})()
    choice = type("Choice", (), {"delta": delta})()
    chunk = type("Chunk", (), {"choices": [choice]})()

    result = core.extract_text_from_chunk(chunk, "chatgpt")
    assert result == "Hello from ChatGPT"


@pytest.mark.legacy
def test_extract_text_from_chunk_fallback():
    """extract_text_from_chunk should delegate to provider and fall back to string"""
    # Test with plain string (fallback case - provider fails, falls back to string)
    with patch("multi_llm_chat.core_modules.legacy_api.create_provider") as mock_create_provider:
        mock_provider = Mock()
        # Provider fails to extract, triggering fallback
        mock_provider.extract_text_from_chunk.side_effect = Exception("extraction failed")
        mock_create_provider.return_value = mock_provider

        chunk = "Plain string chunk"
        result = core.extract_text_from_chunk(chunk, "gemini")
        assert result == "Plain string chunk"

    # Test with invalid chunk (should return empty string)
    with patch("multi_llm_chat.core_modules.legacy_api.create_provider") as mock_create_provider:
        mock_provider = Mock()
        mock_provider.extract_text_from_chunk.side_effect = Exception("extraction failed")
        mock_create_provider.return_value = mock_provider

        chunk = type("Invalid", (), {})()
        result = core.extract_text_from_chunk(chunk, "gemini")
        assert result == ""
