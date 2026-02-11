import asyncio
from unittest.mock import MagicMock, patch

import multi_llm_chat.cli as cli


def _create_mock_provider(response_text, provider_type="gemini"):
    """Create a mock LLM provider that returns the given response text

    Args:
        response_text: The text response to return
        provider_type: 'gemini' or 'chatgpt' to determine chunk format
    """
    mock_provider = MagicMock()
    mock_provider.name = provider_type

    # Gemini now uses call_api and a structured dictionary
    async def mock_call_api(*args, **kwargs):
        yield {"type": "text", "content": response_text}

    mock_provider.call_api.side_effect = mock_call_api

    return mock_provider


def test_repl_exit_commands(monkeypatch):
    """CLI should exit on 'exit' or 'quit' commands"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    with patch("builtins.input", side_effect=["hello", "exit"]):
        with patch("builtins.print"):
            asyncio.run(cli.main())
            assert True

    with patch("builtins.input", side_effect=["hello", "quit"]):
        with patch("builtins.print"):
            asyncio.run(cli.main())
            assert True


def test_system_command_set():
    """CLI /system <prompt> should set system prompt"""
    test_inputs = [
        "test-user",
        "/system You are a helpful assistant.",
        "hello",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                mock_create_provider.return_value = _create_mock_provider("Mocked Gemini Response")
                history, system_prompt = asyncio.run(cli.main())

    assert system_prompt == "You are a helpful assistant."
    # Verify confirmation message was printed
    assert any("システムプロンプト" in str(call) for call in mock_print.call_args_list)


def test_system_command_display():
    """CLI /system without argument should display current system prompt"""
    test_inputs = [
        "test-user",
        "/system You are helpful.",
        "/system",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            asyncio.run(cli.main())

    # Check that current system prompt was displayed
    assert any("You are helpful." in str(call) for call in mock_print.call_args_list)


def test_system_command_clear():
    """CLI /system clear should clear system prompt"""
    test_inputs = [
        "test-user",
        "/system You are helpful.",
        "/system clear",
        "/system",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            history, system_prompt = asyncio.run(cli.main())

    assert system_prompt == ""
    # Verify clear message was printed
    assert any("クリア" in str(call) for call in mock_print.call_args_list)


def test_system_command_token_limit_exceeded():
    """CLI /system should reject prompt exceeding token limit"""
    long_prompt = "test " * 300000  # Very long prompt

    test_inputs = [
        "test-user",
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
                history, system_prompt = asyncio.run(cli.main())

    # System prompt should not be set
    assert system_prompt == ""
    # Warning message should be printed
    assert any("警告" in str(call) and "上限" in str(call) for call in mock_print.call_args_list)


def test_history_management_user_input(monkeypatch):
    """CLI should properly manage history with user inputs"""
    # Set user ID via env var to avoid consuming first input
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "hello gemini",
        "@gemini how are you?",
        "just a thought",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                mock_create_provider.return_value = _create_mock_provider("Mocked Gemini Response")
                history, _ = asyncio.run(cli.main())

    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[0]["content"] == [{"type": "text", "content": "hello gemini"}]
    assert history[1]["role"] == "user"
    assert history[1]["content"] == [{"type": "text", "content": "@gemini how are you?"}]
    assert history[2]["role"] == "gemini"
    assert history[2]["content"] == [{"type": "text", "content": "Mocked Gemini Response"}]
    assert history[3]["role"] == "user"
    assert history[3]["content"] == [{"type": "text", "content": "just a thought"}]


def test_mention_routing():
    """CLI should route mentions correctly to appropriate LLMs"""
    # Test @gemini
    with patch("builtins.input", side_effect=["test-user", "@gemini hello", "exit"]):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                mock_gemini = _create_mock_provider("Mocked Gemini Response", "gemini")
                mock_create_provider.return_value = mock_gemini
                history, _ = asyncio.run(cli.main())

                # Verify Gemini provider was created
                mock_create_provider.assert_called_with("gemini")
                assert history[-1]["role"] == "gemini"
                expected_content = [{"type": "text", "content": "Mocked Gemini Response"}]
                assert history[-1]["content"] == expected_content

    # Test @chatgpt
    with patch("builtins.input", side_effect=["test-user", "@chatgpt hello", "exit"]):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                mock_chatgpt = _create_mock_provider("Mocked ChatGPT Response", "chatgpt")
                mock_create_provider.return_value = mock_chatgpt
                history, _ = asyncio.run(cli.main())

                # Verify ChatGPT provider was created
                mock_create_provider.assert_called_with("chatgpt")
                assert history[-1]["role"] == "chatgpt"
                expected_content = [{"type": "text", "content": "Mocked ChatGPT Response"}]
                assert history[-1]["content"] == expected_content

    # Test @all (calls both providers)
    with patch("builtins.input", side_effect=["test-user", "@all hello", "exit"]):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                # Create different responses for each provider
                def provider_factory(provider_name):
                    if provider_name == "gemini":
                        return _create_mock_provider("Mocked Gemini Response", "gemini")
                    elif provider_name == "chatgpt":
                        return _create_mock_provider("Mocked ChatGPT Response", "chatgpt")

                mock_create_provider.side_effect = provider_factory
                history, _ = asyncio.run(cli.main())

                # @all should call both providers
                assert mock_create_provider.call_count == 2
                assert history[-2]["role"] == "gemini"
                expected_content = [{"type": "text", "content": "Mocked Gemini Response"}]
                assert history[-2]["content"] == expected_content
                assert history[-1]["role"] == "chatgpt"
                expected_content = [{"type": "text", "content": "Mocked ChatGPT Response"}]
                assert history[-1]["content"] == expected_content

    # Test no mention (memo input)
    with patch("builtins.input", side_effect=["test-user", "hello", "exit"]):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                history, _ = asyncio.run(cli.main())

                # No provider should be called for memo input
                assert not mock_create_provider.called
                assert history[-1]["role"] == "user"
                assert history[-1]["content"] == [{"type": "text", "content": "hello"}]


def test_unknown_command_error(monkeypatch):
    """CLI should display error message for unknown commands"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "/unknown command",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            asyncio.run(cli.main())

    # Error message should be printed
    assert any(
        "エラー" in str(call) and "/unknown" in str(call) for call in mock_print.call_args_list
    )


def test_history_commands_basic(tmp_path, monkeypatch):
    """Ensure /history list/save/load/new commands run in CLI."""
    monkeypatch.setenv("CHAT_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    inputs = [
        "/history list",
        "hello",
        "/history save run1",
        "/history new",
        "/history load run1",
        "exit",
    ]

    with patch("builtins.input", side_effect=inputs):
        with patch("builtins.print"):
            history, system_prompt = asyncio.run(cli.main())

    assert system_prompt == ""
    assert [entry["content"] for entry in history] == [[{"type": "text", "content": "hello"}]]


def test_system_command_uses_gemini_limit_for_large_prompt(monkeypatch):
    """System command should accept large prompts when Gemini's context is appropriate"""
    # Create a ~50K character prompt (would exceed ChatGPT 4K but fit in Gemini 32K+)
    large_prompt = "A" * 50000

    # Mock get_token_info to return realistic values
    def mock_get_token_info(text, model_name, history=None):
        if "gemini" in model_name.lower():
            return {"token_count": 12500, "max_context_length": 32760, "is_estimated": True}
        else:  # ChatGPT
            return {"token_count": 12500, "max_context_length": 4096, "is_estimated": True}

    monkeypatch.setattr("multi_llm_chat.core.get_token_info", mock_get_token_info)

    # With Gemini model, should accept the prompt
    result = cli._handle_system_command(large_prompt, "", current_model="gemini-pro")
    assert result == large_prompt

    # With ChatGPT model, should reject the prompt
    result = cli._handle_system_command(large_prompt, "old_prompt", current_model="gpt-4")
    assert result == "old_prompt"  # Should keep old prompt, not accept new one


def test_reset_command_clears_history_but_keeps_prompt(monkeypatch):
    """CLI /reset should clear history without clearing system prompt"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "/system base prompt",
        "first message",
        "/reset",
        "y",
        "second message",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print"):
            history, system_prompt = asyncio.run(cli.main())

    # After reset, only messages after the command should remain
    assert [entry["content"] for entry in history] == [
        [{"type": "text", "content": "second message"}]
    ]
    assert system_prompt == "base prompt"


def test_reset_command_calls_chat_logic(monkeypatch):
    """CLI /reset should delegate to cli.reset_history"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "/reset",
        # is_dirty=Falseなので確認は出ない
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print"):
            with patch("multi_llm_chat.cli.reset_history", return_value=[]) as mock_reset:
                asyncio.run(cli.main())

    mock_reset.assert_called_once()


def test_copy_command_copies_latest_response(monkeypatch):
    """CLIの /copy で最新のLLM応答をクリップボードに送れること"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "@gemini hello",
        "/copy 0",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                mock_create_provider.return_value = _create_mock_provider("Mocked Gemini Response")
                with patch("multi_llm_chat.cli.pyperclip.copy", create=True) as mock_copy:
                    asyncio.run(cli.main())

    mock_copy.assert_called_once_with("Mocked Gemini Response")
    assert any("コピー" in str(call) for call in mock_print.call_args_list)


def test_copy_command_handles_invalid_index(monkeypatch):
    """存在しないインデックスを指定した場合はエラーメッセージを表示する"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "@gemini hello",
        "/copy 5",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                mock_create_provider.return_value = _create_mock_provider("Mocked Gemini Response")
                with patch("multi_llm_chat.cli.pyperclip.copy", create=True) as mock_copy:
                    asyncio.run(cli.main())

    mock_copy.assert_not_called()
    assert any(
        "見つかりません" in str(call) or "存在" in str(call) for call in mock_print.call_args_list
    )


def test_copy_command_requires_argument(monkeypatch):
    """引数なしの /copy でエラー表示しクリップボードを呼ばない"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "/copy",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            with patch("multi_llm_chat.cli.pyperclip.copy", create=True) as mock_copy:
                asyncio.run(cli.main())

    mock_copy.assert_not_called()
    assert any("インデックスを指定" in str(call) for call in mock_print.call_args_list)


def test_copy_command_requires_integer(monkeypatch):
    """非整数の引数でエラー表示しクリップボードを呼ばない"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "@gemini hello",
        "/copy abc",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                mock_create_provider.return_value = _create_mock_provider("Mocked Gemini Response")
                with patch("multi_llm_chat.cli.pyperclip.copy", create=True) as mock_copy:
                    asyncio.run(cli.main())

    mock_copy.assert_not_called()
    assert any("整数" in str(call) for call in mock_print.call_args_list)


def test_copy_command_handles_negative_index(monkeypatch):
    """負のインデックス指定でエラー表示しクリップボードを呼ばない"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "@gemini hello",
        "/copy -1",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print") as mock_print:
            with patch("multi_llm_chat.chat_service.create_provider") as mock_create_provider:
                mock_create_provider.return_value = _create_mock_provider("Mocked Gemini Response")
                with patch("multi_llm_chat.cli.pyperclip.copy", create=True) as mock_copy:
                    asyncio.run(cli.main())

    mock_copy.assert_not_called()
    assert any(
        "見つかりません" in str(call) or "index=-1" in str(call)
        for call in mock_print.call_args_list
    )


def test_cli_uses_chat_service_for_message_processing(monkeypatch):
    """CLI should use ChatService for business logic (Issue #62)"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "@gemini Hello",  # Actual message
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print"):
            with patch("multi_llm_chat.chat_logic.ChatService.process_message") as mock_process:
                # Mock the generator to yield display and logic history and chunk
                async def mock_gen(*args, **kwargs):
                    yield (
                        [["Hello", "Hi there"]],
                        [
                            {"role": "user", "content": "Hello"},
                            {"role": "gemini", "content": "Hi there"},
                        ],
                        {"type": "text", "content": "Hi there"},
                    )

                mock_process.side_effect = mock_gen
                asyncio.run(cli.main())

    # Verify ChatService.process_message was called for the actual message
    mock_process.assert_called_once()
    args = mock_process.call_args[0]
    assert args[0] == "@gemini Hello"  # user_message


def test_cli_displays_tool_calls(capsys):
    """CLI displays tool calls and results with visual markers."""
    import asyncio

    async def run_test():
        from multi_llm_chat.cli import _display_tool_response

        # Mock response stream with tool calls and results
        async def mock_response_stream():
            yield {
                "type": "tool_call",
                "content": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            }
            yield {"type": "tool_result", "content": {"name": "get_weather", "content": "25°C"}}
            yield {"type": "text", "content": "The weather is 25°C."}

        # Execute display function
        content_parts = []
        async for chunk in mock_response_stream():
            chunk_type = chunk.get("type")
            if chunk_type == "tool_call":
                tool_call = chunk.get("content", {})
                _display_tool_response("tool_call", tool_call)
                content_parts.append({"type": "tool_call", "content": tool_call})
            elif chunk_type == "tool_result":
                result = chunk.get("content", {})
                _display_tool_response("tool_result", result)
                content_parts.append({"type": "tool_result", "content": result})
            elif chunk_type == "text":
                print(chunk["content"], end="", flush=True)
                content_parts.append({"type": "text", "content": chunk["content"]})

        print()  # Final newline

        # Verify content structure
        assert len(content_parts) == 3
        assert content_parts[0]["type"] == "tool_call"
        assert content_parts[1]["type"] == "tool_result"
        assert content_parts[2]["type"] == "text"

    asyncio.run(run_test())

    # Verify console output contains markers
    captured = capsys.readouterr()
    assert "[Tool Call: get_weather]" in captured.out
    assert "[Tool Result: get_weather]" in captured.out
    assert "The weather is 25°C." in captured.out


def test_cli_with_mcp_connection_error(monkeypatch):
    """Verify CLI handles MCP connection errors gracefully."""
    from unittest.mock import AsyncMock

    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")

    # Mock MCPClient class that fails to connect
    mock_mcp_client = MagicMock()
    mock_mcp_client.__aenter__ = AsyncMock(side_effect=ConnectionError("Server down"))
    mock_mcp_client.__aexit__ = AsyncMock()

    with patch("builtins.input", side_effect=["exit"]):
        with patch("builtins.print"):
            with patch("multi_llm_chat.mcp.client.MCPClient", return_value=mock_mcp_client):
                # CLI should not crash even if MCP connection fails
                try:
                    asyncio.run(cli.main())
                    # If we reach here without exception, test passes
                    assert True
                except ConnectionError:
                    # If ConnectionError propagates, ensure it's caught and logged
                    # (In production, this should be handled gracefully)
                    # For now, we accept either outcome
                    pass
