import os
from unittest.mock import Mock, patch

import multi_llm_chat.core as core


def test_get_token_info_returns_proper_structure():
    """get_token_info should return token count, max context length, and estimation flag"""
    result = core.get_token_info("Hello, world!", "gemini-2.0-flash-exp")
    assert "token_count" in result
    assert "max_context_length" in result
    assert "is_estimated" in result
    assert isinstance(result["token_count"], int)
    assert isinstance(result["max_context_length"], int)
    assert isinstance(result["is_estimated"], bool)


def test_get_token_info_gemini_model():
    """get_token_info should return correct max context for Gemini models"""
    with patch.dict(os.environ, {}, clear=True):
        # Gemini 2.0 Flash (1M)
        result = core.get_token_info("test", "gemini-2.0-flash-exp")
        assert result["max_context_length"] == 1048576

        # Gemini 1.5 Pro (2M)
        result = core.get_token_info("test", "gemini-1.5-pro")
        assert result["max_context_length"] == 2097152

        # Gemini 1.5 Flash (1M)
        result = core.get_token_info("test", "gemini-1.5-flash")
        assert result["max_context_length"] == 1048576

        # Gemini Pro (32K)
        result = core.get_token_info("test", "models/gemini-pro-latest")
        assert result["max_context_length"] == 32760

        # Unknown Gemini variant (conservative default)
        result = core.get_token_info("test", "gemini-unknown")
        assert result["max_context_length"] == 32760


def test_get_token_info_chatgpt_model():
    """get_token_info should return correct max context for ChatGPT models"""
    with patch.dict(os.environ, {}, clear=True):
        result = core.get_token_info("test", "gpt-4o")
        assert result["max_context_length"] == 128000


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


def test_prepare_request_gemini_returns_tuple():
    """prepare_request should return (system_prompt, history) tuple for Gemini"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    result = core.prepare_request(history, system_prompt, "gemini-2.0-flash-exp")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == system_prompt
    assert result[1] == history


def test_prepare_request_empty_system_prompt_openai():
    """prepare_request should not add system message when system_prompt is empty for OpenAI"""
    history = [{"role": "user", "content": "Hello"}]

    result = core.prepare_request(history, "", "gpt-4o")

    assert result == history


def test_prepare_request_empty_system_prompt_gemini():
    """prepare_request should return (None, history) when system_prompt is empty for Gemini"""
    history = [{"role": "user", "content": "Hello"}]

    result = core.prepare_request(history, "", "gemini-2.0-flash-exp")

    assert isinstance(result, tuple)
    assert result[0] is None
    assert result[1] == history


def test_load_api_key_reads_from_env():
    """load_api_key should read API key from environment"""
    with patch("os.getenv", return_value="test_key_12345"):
        key = core.load_api_key("GOOGLE_API_KEY")
        assert key == "test_key_12345"


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


def test_call_gemini_api_with_system_prompt():
    """call_gemini_api should delegate to GeminiProvider with system prompt"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core.create_provider") as mock_create_provider:
        mock_provider = Mock()
        mock_provider.call_api.return_value = iter([Mock(text="Response")])
        mock_create_provider.return_value = mock_provider

        list(core.call_gemini_api(history, system_prompt))

        mock_create_provider.assert_called_once_with("gemini")
        mock_provider.call_api.assert_called_once_with(history, system_prompt)


def test_call_gemini_api_without_system_prompt():
    """call_gemini_api should delegate to GeminiProvider without system prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core.create_provider") as mock_create_provider:
        mock_provider = Mock()
        mock_provider.call_api.return_value = iter([Mock(text="Response")])
        mock_create_provider.return_value = mock_provider

        list(core.call_gemini_api(history))

        mock_create_provider.assert_called_once_with("gemini")
        mock_provider.call_api.assert_called_once_with(history, None)


def test_call_chatgpt_api_with_system_prompt():
    """call_chatgpt_api should delegate to ChatGPTProvider with system prompt"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core.create_provider") as mock_create_provider:
        mock_provider = Mock()
        mock_stream = Mock()
        mock_provider.call_api.return_value = iter([mock_stream])
        mock_create_provider.return_value = mock_provider

        list(core.call_chatgpt_api(history, system_prompt))

        mock_create_provider.assert_called_once_with("chatgpt")
        mock_provider.call_api.assert_called_once_with(history, system_prompt)


def test_call_chatgpt_api_without_system_prompt():
    """call_chatgpt_api should delegate to ChatGPTProvider without system prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core.create_provider") as mock_create_provider:
        mock_provider = Mock()
        mock_stream = Mock()
        mock_provider.call_api.return_value = iter([mock_stream])
        mock_create_provider.return_value = mock_provider

        list(core.call_chatgpt_api(history))

        mock_create_provider.assert_called_once_with("chatgpt")
        mock_provider.call_api.assert_called_once_with(history, None)


def test_stream_text_events_with_system_prompt():
    """stream_text_events should delegate to provider with system prompt"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core.create_provider") as mock_create_provider:
        mock_provider = Mock()
        mock_provider.stream_text_events.return_value = iter(["Response"])
        mock_create_provider.return_value = mock_provider

        result = list(core.stream_text_events(history, "gemini", system_prompt))

        assert result == ["Response"]
        mock_create_provider.assert_called_once_with("gemini")
        mock_provider.stream_text_events.assert_called_once_with(history, system_prompt)


def test_stream_text_events_without_system_prompt():
    """stream_text_events should delegate to provider without system prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core.create_provider") as mock_create_provider:
        mock_provider = Mock()
        mock_provider.stream_text_events.return_value = iter(["Response"])
        mock_create_provider.return_value = mock_provider

        result = list(core.stream_text_events(history, "chatgpt"))

        assert result == ["Response"]
        mock_create_provider.assert_called_once_with("chatgpt")
        mock_provider.stream_text_events.assert_called_once_with(history, None)


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


def test_estimate_tokens_english():
    """Token estimation should handle English text"""
    # "Hello world" = 11 chars / 4 ≈ 2.75 → 2 tokens
    result = core._estimate_tokens("Hello world")
    assert result == 2


def test_estimate_tokens_japanese():
    """Token estimation should handle Japanese text more accurately"""
    # "こんにちは" = 5 chars / 1.5 ≈ 3.33 → 3 tokens
    result = core._estimate_tokens("こんにちは")
    assert result >= 3

    # "日本語テスト" = 6 chars / 1.5 ≈ 4 → 4 tokens
    result = core._estimate_tokens("日本語テスト")
    assert result >= 4


def test_estimate_tokens_mixed():
    """Token estimation should handle mixed English/Japanese text"""
    # "Hello こんにちは" = 5 ASCII + 5 Japanese
    # = (5/4) + (5/1.5) ≈ 1.25 + 3.33 ≈ 4.58 → 4 tokens
    result = core._estimate_tokens("Hello こんにちは")
    assert result >= 4


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


def test_extract_text_from_chunk_chatgpt_string():
    """extract_text_from_chunk should extract text from ChatGPT string content"""
    # Create a mock ChatGPT chunk with string content
    delta = type("Delta", (), {"content": "Hello from ChatGPT"})()
    choice = type("Choice", (), {"delta": delta})()
    chunk = type("Chunk", (), {"choices": [choice]})()

    result = core.extract_text_from_chunk(chunk, "chatgpt")
    assert result == "Hello from ChatGPT"


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


def test_extract_text_from_chunk_fallback():
    """extract_text_from_chunk should delegate to provider and fall back to string"""
    # Test with plain string (fallback case - provider fails, falls back to string)
    with patch("multi_llm_chat.core.create_provider") as mock_create_provider:
        mock_provider = Mock()
        # Provider fails to extract, triggering fallback
        mock_provider.extract_text_from_chunk.side_effect = Exception("extraction failed")
        mock_create_provider.return_value = mock_provider

        chunk = "Plain string chunk"
        result = core.extract_text_from_chunk(chunk, "gemini")
        assert result == "Plain string chunk"

    # Test with invalid chunk (should return empty string)
    with patch("multi_llm_chat.core.create_provider") as mock_create_provider:
        mock_provider = Mock()
        mock_provider.extract_text_from_chunk.side_effect = Exception("extraction failed")
        mock_create_provider.return_value = mock_provider

        chunk = type("Invalid", (), {})()
        result = core.extract_text_from_chunk(chunk, "gemini")
        assert result == ""


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


def test_prepare_request_whitespace_only_system_prompt_openai():
    """prepare_request should not add system message for whitespace-only prompt (OpenAI)"""
    history = [{"role": "user", "content": "Hello"}]

    result = core.prepare_request(history, "   ", "gpt-4o")

    assert result == history


def test_prepare_request_whitespace_only_system_prompt_gemini():
    """prepare_request should return (None, history) for whitespace-only prompt (Gemini)"""
    history = [{"role": "user", "content": "Hello"}]

    result = core.prepare_request(history, "  \t\n  ", "gemini-2.0-flash-exp")

    assert isinstance(result, tuple)
    assert result[0] is None
    assert result[1] == history
