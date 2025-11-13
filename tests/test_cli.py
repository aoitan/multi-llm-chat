from unittest.mock import patch

import multi_llm_chat.cli as cli


def _gemini_stream(*_args, **_kwargs):
    return iter(["Mocked Gemini Response"])


def _chatgpt_stream(*_args, **_kwargs):
    return iter(["Mocked ChatGPT Response"])


def test_repl_exit_commands():
    """CLI should exit on 'exit' or 'quit' commands"""
    with patch("builtins.input", side_effect=["hello", "exit"]):
        with patch("builtins.print"):
            cli.main()
            assert True

    with patch("builtins.input", side_effect=["hello", "quit"]):
        with patch("builtins.print"):
            cli.main()
            assert True


def test_system_command_set():
    """CLI /system <prompt> should set system prompt"""
    test_inputs = [
        "/system You are a helpful assistant.",
        "hello",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            with patch("multi_llm_chat.core.call_gemini_api", side_effect=_gemini_stream):
                history, system_prompt = cli.main()

    assert system_prompt == "You are a helpful assistant."
    # Verify confirmation message was printed
    assert any("システムプロンプト" in str(call) for call in mock_print.call_args_list)


def test_system_command_display():
    """CLI /system without argument should display current system prompt"""
    test_inputs = [
        "/system You are helpful.",
        "/system",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            cli.main()

    # Check that current system prompt was displayed
    assert any("You are helpful." in str(call) for call in mock_print.call_args_list)


def test_system_command_clear():
    """CLI /system clear should clear system prompt"""
    test_inputs = [
        "/system You are helpful.",
        "/system clear",
        "/system",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            history, system_prompt = cli.main()

    assert system_prompt == ""
    # Verify clear message was printed
    assert any("クリア" in str(call) for call in mock_print.call_args_list)


def test_system_command_token_limit_exceeded():
    """CLI /system should reject prompt exceeding token limit"""
    long_prompt = "test " * 300000  # Very long prompt

    test_inputs = [
        f"/system {long_prompt}",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
                mock_token_info.return_value = {
                    "token_count": 2000000,
                    "max_context_length": 1048576,
                    "is_estimated": False,
                }
                history, system_prompt = cli.main()

    # System prompt should not be set
    assert system_prompt == ""
    # Warning message should be printed
    assert any("警告" in str(call) and "上限" in str(call) for call in mock_print.call_args_list)


def test_history_management_user_input():
    """CLI should properly manage history with user inputs"""
    test_inputs = ["hello gemini", "@gemini how are you?", "just a thought", "exit"]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print"):
            with patch("multi_llm_chat.core.call_gemini_api", side_effect=_gemini_stream):
                with patch("multi_llm_chat.core.call_chatgpt_api", side_effect=_chatgpt_stream):
                    history, _ = cli.main()

    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello gemini"
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "@gemini how are you?"
    assert history[2]["role"] == "gemini"
    assert history[2]["content"] == "Mocked Gemini Response"
    assert history[3]["role"] == "user"
    assert history[3]["content"] == "just a thought"


@patch("multi_llm_chat.core.call_gemini_api", side_effect=_gemini_stream)
@patch("multi_llm_chat.core.call_chatgpt_api", side_effect=_chatgpt_stream)
def test_mention_routing(mock_chatgpt_api, mock_gemini_api):
    """CLI should route mentions correctly to appropriate LLMs"""
    # Test @gemini
    with patch("builtins.input", side_effect=["@gemini hello", "exit"]):
        with patch("builtins.print"):
            history, _ = cli.main()
            assert mock_gemini_api.called
            assert not mock_chatgpt_api.called
            assert history[-1]["role"] == "gemini"
            assert history[-1]["content"] == "Mocked Gemini Response"
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test @chatgpt
    with patch("builtins.input", side_effect=["@chatgpt hello", "exit"]):
        with patch("builtins.print"):
            history, _ = cli.main()
            assert not mock_gemini_api.called
            assert mock_chatgpt_api.called
            assert history[-1]["role"] == "chatgpt"
            assert history[-1]["content"] == "Mocked ChatGPT Response"
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test @all
    with patch("builtins.input", side_effect=["@all hello", "exit"]):
        with patch("builtins.print"):
            history_snapshots = {}

            def gemini_capture(history, system_prompt=None):
                history_snapshots["gemini"] = [entry.copy() for entry in history]
                return iter(["Mocked Gemini Response"])

            def chatgpt_capture(history, system_prompt=None):
                history_snapshots["chatgpt"] = [entry.copy() for entry in history]
                return iter(["Mocked ChatGPT Response"])

            mock_gemini_api.side_effect = gemini_capture
            mock_chatgpt_api.side_effect = chatgpt_capture

            history, _ = cli.main()
            assert mock_gemini_api.called
            assert mock_chatgpt_api.called
            assert history[-2]["role"] == "gemini"
            assert history[-2]["content"] == "Mocked Gemini Response"
            assert history[-1]["role"] == "chatgpt"
            assert history[-1]["content"] == "Mocked ChatGPT Response"
            assert history_snapshots["gemini"] == history_snapshots["chatgpt"]
            assert all(entry["role"] != "gemini" for entry in history_snapshots["chatgpt"])

            mock_gemini_api.side_effect = _gemini_stream
            mock_chatgpt_api.side_effect = _chatgpt_stream
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test no mention
    with patch("builtins.input", side_effect=["hello", "exit"]):
        with patch("builtins.print"):
            history, _ = cli.main()
            assert not mock_gemini_api.called
            assert not mock_chatgpt_api.called
            assert history[-1]["role"] == "user"
            assert history[-1]["content"] == "hello"


def test_unknown_command_error():
    """CLI should display error message for unknown commands"""
    test_inputs = [
        "/unknown command",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            cli.main()

    # Error message should be printed
    assert any(
        "エラー" in str(call) and "/unknown" in str(call) for call in mock_print.call_args_list
    )
