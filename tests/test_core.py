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
    result = core.get_token_info("test", "gemini-2.0-flash-exp")
    assert result["max_context_length"] == 1048576


def test_get_token_info_chatgpt_model():
    """get_token_info should return correct max context for ChatGPT models"""
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
    """call_gemini_api should pass system_prompt to GenerativeModel"""
    history = [{"role": "user", "content": "Hello"}]
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core.genai.GenerativeModel") as mock_model_class:
        with patch("multi_llm_chat.core._configure_gemini", return_value=True):
            mock_instance = Mock()
            mock_instance.generate_content.return_value = iter([Mock(text="Response")])
            mock_model_class.return_value = mock_instance

            list(core.call_gemini_api(history, system_prompt))

            mock_model_class.assert_called_once_with(
                core.GEMINI_MODEL, system_instruction=system_prompt
            )


def test_call_gemini_api_without_system_prompt():
    """call_gemini_api should use cached model when no system_prompt"""
    history = [{"role": "user", "content": "Hello"}]

    with patch("multi_llm_chat.core._get_gemini_model") as mock_get_model:
        mock_model = Mock()
        mock_model.generate_content.return_value = iter([Mock(text="Response")])
        mock_get_model.return_value = mock_model

        list(core.call_gemini_api(history))

        mock_get_model.assert_called_once()


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
