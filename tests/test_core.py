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
    assert result[0]["parts"] == ["Hello"]
    assert result[1]["role"] == "model"
    assert result[1]["parts"] == ["Hi there"]


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
    """call_gemini_api should pass system_prompt to cached model"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core._get_gemini_model") as mock_get_model:
        mock_instance = Mock()
        mock_instance.generate_content.return_value = iter([Mock(text="Response")])
        mock_get_model.return_value = mock_instance

        list(core.call_gemini_api(history, system_prompt))

        mock_get_model.assert_called_once_with(system_prompt)


def test_call_gemini_api_without_system_prompt():
    """call_gemini_api should use cached model when no system_prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core._get_gemini_model") as mock_get_model:
        mock_model = Mock()
        mock_model.generate_content.return_value = iter([Mock(text="Response")])
        mock_get_model.return_value = mock_model

        list(core.call_gemini_api(history))

        mock_get_model.assert_called_once_with(None)


def test_call_chatgpt_api_with_system_prompt():
    """call_chatgpt_api should prepend system message to history"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core._get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_stream = Mock()
        mock_stream.model_dump.return_value = {}
        mock_client.chat.completions.create.return_value = iter([mock_stream])
        mock_get_client.return_value = mock_client

        list(core.call_chatgpt_api(history, system_prompt))

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[1]["role"] == "user"


def test_call_chatgpt_api_without_system_prompt():
    """call_chatgpt_api should not add system message when no system_prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core._get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_stream = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_stream])
        mock_get_client.return_value = mock_client

        list(core.call_chatgpt_api(history))

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "user"
        assert len(messages) == 1


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


def test_hash_prompt():
    """Prompt hashing should be consistent and collision-resistant"""
    prompt1 = "You are a helpful assistant."
    prompt2 = "You are a helpful assistant."
    prompt3 = "You are a different assistant."

    hash1 = core._hash_prompt(prompt1)
    hash2 = core._hash_prompt(prompt2)
    hash3 = core._hash_prompt(prompt3)

    # Same prompts should have same hash
    assert hash1 == hash2

    # Different prompts should have different hash
    assert hash1 != hash3

    # Hash should be hex string
    assert len(hash1) == 64  # SHA256 = 64 hex chars
    assert all(c in "0123456789abcdef" for c in hash1)


def test_get_gemini_model_cache_with_hash():
    """Gemini model cache should use hash keys to prevent memory leak"""
    with patch("multi_llm_chat.core._configure_gemini", return_value=True):
        with patch("multi_llm_chat.core.genai.GenerativeModel"):
            # Clear cache
            core._gemini_models_cache.clear()

            # Create model with very long system prompt
            long_prompt = "A" * 10000  # 10K characters
            model1 = core._get_gemini_model(long_prompt)

            # Cache should contain hash, not full prompt
            assert len(core._gemini_models_cache) == 1
            cache_key = list(core._gemini_models_cache.keys())[0]

            # Key should be hash (64 hex chars), not the long prompt
            assert len(cache_key) == 64
            assert cache_key != long_prompt

            # Same prompt should hit cache
            model2 = core._get_gemini_model(long_prompt)
            assert model1 == model2
            assert len(core._gemini_models_cache) == 1


def test_extract_text_from_chunk_gemini():
    """extract_text_from_chunk should extract text from Gemini chunk"""
    # Create a mock Gemini chunk
    chunk = type("Chunk", (), {"text": "Hello from Gemini"})()
    result = core.extract_text_from_chunk(chunk, "gemini")
    assert result == "Hello from Gemini"


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
    """extract_text_from_chunk should fall back to string representation"""
    # Test with plain string
    chunk = "Plain string chunk"
    result = core.extract_text_from_chunk(chunk, "gemini")
    assert result == "Plain string chunk"

    # Test with invalid chunk (should return empty string)
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
    assert result[0] == {"role": "user", "parts": ["Hello"]}
    assert result[1] == {"role": "user", "parts": ["Another question"]}
    assert result[2] == {"role": "model", "parts": ["Answer from Gemini"]}


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


@patch("multi_llm_chat.core.genai")
@patch("multi_llm_chat.core._configure_gemini")
def test_get_gemini_model_with_empty_system_prompt(mock_configure, mock_genai):
    """_get_gemini_model should use default model for empty system prompt"""
    mock_configure.return_value = True
    mock_model = Mock()
    mock_genai.GenerativeModel.return_value = mock_model

    # Reset cached model
    core._gemini_model = None

    result = core._get_gemini_model("")

    # Should create model without system_instruction
    mock_genai.GenerativeModel.assert_called_once_with(core.GEMINI_MODEL)
    assert result == mock_model


@patch("multi_llm_chat.core.genai")
@patch("multi_llm_chat.core._configure_gemini")
def test_get_gemini_model_with_whitespace_system_prompt(mock_configure, mock_genai):
    """_get_gemini_model should use default model for whitespace-only system prompt"""
    mock_configure.return_value = True
    mock_model = Mock()
    mock_genai.GenerativeModel.return_value = mock_model

    # Reset cached model
    core._gemini_model = None

    result = core._get_gemini_model("  \t\n  ")

    # Should create model without system_instruction
    mock_genai.GenerativeModel.assert_called_once_with(core.GEMINI_MODEL)
    assert result == mock_model
