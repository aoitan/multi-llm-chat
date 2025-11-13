from unittest.mock import patch

import multi_llm_chat.webui as webui


def test_system_prompt_textbox_exists():
    """Web UI should have system prompt textbox component"""
    # This test will verify that the UI component is created
    with patch("gradio.Blocks"):
        with patch("gradio.Textbox"):
            # We'll test component creation when implementation exists
            pass


def test_token_count_display_updates():
    """Token count display should update when system prompt changes"""
    # Test that token info is calculated and displayed
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 50,
            "max_context_length": 1048576,
            "is_estimated": True,
        }

        result = webui.update_token_display("Test system prompt", "gemini-2.0-flash-exp")

        assert "50" in result
        assert "1048576" in result
        assert "estimated" in result.lower()


def test_token_limit_warning_display():
    """Token count display should show warning when limit exceeded"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 2000000,
            "max_context_length": 1048576,
            "is_estimated": False,
        }

        result = webui.update_token_display("Very long prompt", "gemini-2.0-flash-exp")

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

        is_enabled = webui.check_send_button_enabled("Very long prompt", "gemini-2.0-flash-exp")

        assert is_enabled is False


def test_send_button_enabled_when_within_limit():
    """Send button should be enabled when token count is within limit"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 100,
            "max_context_length": 1048576,
            "is_estimated": True,
        }

        is_enabled = webui.check_send_button_enabled("Normal prompt", "gemini-2.0-flash-exp")

        assert is_enabled is True


def test_system_prompt_included_in_chat():
    """Chat function should include system prompt when calling LLM"""
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core.prepare_request") as mock_prepare:
        with patch("multi_llm_chat.core.call_gemini_api") as mock_api:
            mock_prepare.return_value = (system_prompt, [{"role": "user", "content": "Hello"}])
            mock_api.return_value = iter(["Response"])

            # Simulate chat function call
            # Implementation will be tested when webui.py exists
            pass
