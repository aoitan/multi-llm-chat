from unittest.mock import MagicMock, patch

import pytest

import multi_llm_chat.chat_logic as chat_logic


def _create_mock_provider(response_text, provider_type="gemini"):
    """Create a mock LLM provider that returns the given response text

    Args:
        response_text: The text response to return
        provider_type: 'gemini' or 'chatgpt' to determine chunk format
    """
    mock_provider = MagicMock()

    if provider_type == "gemini":
        # Gemini now uses call_api and a structured dictionary
        mock_provider.call_api.return_value = iter([{"type": "text", "content": response_text}])
    else:  # chatgpt
        mock_provider.stream_text_events.return_value = iter([response_text])

    return mock_provider


def test_repl_exit_commands():
    with patch("builtins.input", side_effect=["test-user", "hello", "exit"]):
        with patch("builtins.print"):
            chat_logic.main()
            assert True

    with patch("builtins.input", side_effect=["test-user", "hello", "quit"]):
        with patch("builtins.print"):
            chat_logic.main()
            assert True


def test_history_management_user_input():
    test_inputs = ["hello gemini", "@gemini how are you?", "just a thought"]

    with patch("builtins.input", side_effect=test_inputs + ["exit"]):
        with patch("builtins.print"):  # Mock print to avoid console output
            # Mock API calls to control history length
            with patch("multi_llm_chat.chat_logic.create_provider") as mock_create_provider:
                mock_create_provider.return_value = _create_mock_provider("Mocked Gemini Response")
                history = chat_logic.main()

            # Expected history: user, user, gemini, user
            assert len(history) == 4
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "hello gemini"
            assert history[1]["role"] == "user"
            assert history[1]["content"] == "@gemini how are you?"
            assert history[2]["role"] == "gemini"
            assert history[2]["content"] == [{"type": "text", "content": "Mocked Gemini Response"}]
            assert history[3]["role"] == "user"
            assert history[3]["content"] == "just a thought"


def test_mention_routing():
    """CLI should route mentions correctly to appropriate LLMs"""
    # Test @gemini
    with patch("builtins.input", side_effect=["test-user", "@gemini hello", "exit"]):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_logic.create_provider") as mock_create_provider:
                mock_gemini = _create_mock_provider("Mocked Gemini Response", "gemini")
                mock_chatgpt = _create_mock_provider("Mocked ChatGPT Response", "chatgpt")

                def provider_factory(provider_name):
                    if provider_name == "gemini":
                        return mock_gemini
                    elif provider_name == "chatgpt":
                        return mock_chatgpt

                mock_create_provider.side_effect = provider_factory
                history = chat_logic.main()

                # Verify Gemini provider was created
                assert any(call[0][0] == "gemini" for call in mock_create_provider.call_args_list)
                assert history[-1]["role"] == "gemini"
                assert history[-1]["content"] == [{"type": "text", "content": "Mocked Gemini Response"}]

    # Test @chatgpt
    with patch("builtins.input", side_effect=["test-user", "@chatgpt hello", "exit"]):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_logic.create_provider") as mock_create_provider:
                mock_gemini = _create_mock_provider("Mocked Gemini Response", "gemini")
                mock_chatgpt = _create_mock_provider("Mocked ChatGPT Response", "chatgpt")

                def provider_factory(provider_name):
                    if provider_name == "gemini":
                        return mock_gemini
                    elif provider_name == "chatgpt":
                        return mock_chatgpt

                mock_create_provider.side_effect = provider_factory
                history = chat_logic.main()

                # Verify ChatGPT provider was created
                assert any(call[0][0] == "chatgpt" for call in mock_create_provider.call_args_list)
                assert history[-1]["role"] == "chatgpt"
                assert history[-1]["content"] == "Mocked ChatGPT Response"

    # Test @all (calls both providers)
    with patch("builtins.input", side_effect=["test-user", "@all hello", "exit"]):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_logic.create_provider") as mock_create_provider:
                # Create different responses for each provider
                def provider_factory(provider_name):
                    if provider_name == "gemini":
                        return _create_mock_provider("Mocked Gemini Response", "gemini")
                    elif provider_name == "chatgpt":
                        return _create_mock_provider("Mocked ChatGPT Response", "chatgpt")

                mock_create_provider.side_effect = provider_factory
                history = chat_logic.main()

                # @all should call both providers (2 for ChatService init)
                # ChatService creates both providers on initialization
                assert mock_create_provider.call_count == 2
                assert history[-2]["role"] == "gemini"
                assert history[-2]["content"] == [{"type": "text", "content": "Mocked Gemini Response"}]
                assert history[-1]["role"] == "chatgpt"
                assert history[-1]["content"] == "Mocked ChatGPT Response"


def test_reset_command_clears_history(monkeypatch):
    """chat_logic.main should honor /reset and return only post-reset messages"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "/system base prompt",
        "before reset",
        "/reset",
        "y",
        "after reset",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print"):
            history = chat_logic.main()

    assert [entry["content"] for entry in history] == ["after reset"]


def test_get_llm_response_by_index():
    """指定インデックスのLLM応答を取得できること"""
    history = [
        {"role": "user", "content": "hi"},
        {"role": "gemini", "content": "G-1"},
        {"role": "user", "content": "hello"},
        {"role": "chatgpt", "content": "C-1"},
    ]

    assert chat_logic.get_llm_response(history, 0) == "C-1"
    assert chat_logic.get_llm_response(history, 1) == "G-1"


def test_get_llm_response_raises_on_missing():
    """LLM応答が存在しない場合はIndexErrorとなること"""
    history = [{"role": "user", "content": "only user"}]

    with pytest.raises(IndexError):
        chat_logic.get_llm_response(history, 0)


def test_get_llm_response_raises_on_negative_index():
    """負のインデックス指定でIndexErrorとなること"""
    history = [{"role": "gemini", "content": "ok"}]

    with pytest.raises(IndexError):
        chat_logic.get_llm_response(history, -1)
