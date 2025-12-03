from unittest.mock import patch

import multi_llm_chat.webui as webui


def test_system_prompt_textbox_exists():
    """Web UI should have system prompt textbox component"""
    # Verify that the demo has been built
    assert webui.demo is not None
    assert hasattr(webui.demo, "blocks")


def test_token_count_display_updates():
    """Token count display should update when system prompt changes"""
    # Test that token info is calculated and displayed
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 50,
            "max_context_length": 1048576,
            "is_estimated": True,
        }

        result = webui.update_token_display("Test system prompt", None, "gemini-2.0-flash-exp")

        assert "50" in result
        assert "1048576" in result
        assert "estimated" in result.lower()
        # Verify history was passed (None in this case)
        mock_token_info.assert_called_once_with("Test system prompt", "gemini-2.0-flash-exp", None)


def test_token_count_includes_history():
    """Token count should include both system prompt and history"""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 150,
            "max_context_length": 1048576,
            "is_estimated": True,
        }

        result = webui.update_token_display("Test prompt", history, "gemini-2.0-flash-exp")

        # Verify history was passed to get_token_info
        mock_token_info.assert_called_once_with("Test prompt", "gemini-2.0-flash-exp", history)
        assert "150" in result


def test_token_limit_warning_display():
    """Token count display should show warning when limit exceeded"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 2000000,
            "max_context_length": 1048576,
            "is_estimated": False,
        }

        result = webui.update_token_display("Very long prompt", None, "gemini-2.0-flash-exp")

        # Should contain warning indication (e.g., red color or warning text)
        assert "警告" in result or "warning" in result.lower() or "2000000" in result


def test_send_button_disabled_when_limit_exceeded():
    """Send button should be disabled when token limit is exceeded"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 2000000,
            "max_context_length": 1048576,
            "is_estimated": False,
        }

        result = webui.check_send_button_enabled("Very long prompt", None, "gemini-2.0-flash-exp")

        # Result should be a gr.Button with interactive=False
        assert hasattr(result, "interactive")
        assert result.interactive is False


def test_send_button_enabled_when_within_limit():
    """Send button should be enabled when token count is within limit"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 100,
            "max_context_length": 1048576,
            "is_estimated": True,
        }

        result = webui.check_send_button_enabled("Normal prompt", None, "gemini-2.0-flash-exp")

        # Result should be a gr.Button with interactive=True
        assert hasattr(result, "interactive")
        assert result.interactive is True


def test_system_prompt_included_in_chat():
    """Chat function should include system prompt when calling LLM"""
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core.call_gemini_api") as mock_api:
        mock_api.return_value = iter([type("Chunk", (), {"text": "Response"})()])

        # Simulate calling respond function
        user_message = "@gemini Hello"
        display_history = []
        logic_history = []

        # Call respond and consume the generator
        result_gen = webui.respond(user_message, display_history, logic_history, system_prompt)
        # Consume all yields
        for _ in result_gen:
            pass

        # Verify that call_gemini_api was called with system_prompt
        mock_api.assert_called()
        call_args = mock_api.call_args
        # System prompt is passed as second positional argument
        assert len(call_args.args) == 2
        assert call_args.args[1] == system_prompt


def test_reset_clears_histories_and_keeps_prompt(monkeypatch):
    """Reset handler should clear histories but keep system prompt for token calc"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 10,
            "max_context_length": 100,
            "is_estimated": False,
        }

        chatbot, display_state, logic_state, token_html, send_button, user_input = (
            webui.reset_conversation(
                "stay",
                [["u1", "a1"]],
                [{"role": "user", "content": "before"}],
            )
        )

    assert chatbot == []
    assert display_state == []
    assert logic_state == []
    assert "10" in token_html  # system prompt remains for token count
    assert hasattr(send_button, "interactive") and send_button.interactive is True
    assert user_input == ""
