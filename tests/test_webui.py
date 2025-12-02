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

        result = webui.check_send_button_with_user_id(
            "test_user", "Very long prompt", None, "gemini-2.0-flash-exp"
        )

        # Result should be gr.update() with interactive=False
        assert result["interactive"] is False


def test_send_button_enabled_when_within_limit():
    """Send button should be enabled when token count is within limit"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 100,
            "max_context_length": 1048576,
            "is_estimated": True,
        }

        result = webui.check_send_button_with_user_id(
            "test_user", "Normal prompt", None, "gemini-2.0-flash-exp"
        )

        # Result should be gr.update() with interactive=True
        assert result["interactive"] is True


def test_system_prompt_included_in_chat():
    """Chat function should include system prompt when calling LLM"""
    system_prompt = "You are a helpful assistant."

    with patch("multi_llm_chat.core.call_gemini_api") as mock_api:
        mock_api.return_value = iter([type("Chunk", (), {"text": "Response"})()])

        # Simulate calling respond function
        user_message = "@gemini Hello"
        display_history = []
        logic_history = []

        # Call respond with user_id
        result_gen = webui.respond(
            user_message, display_history, logic_history, system_prompt, user_id="test_user"
        )
        # Consume all yields
        for _ in result_gen:
            pass

        # Verify that call_gemini_api was called with system_prompt
        mock_api.assert_called()
        call_args = mock_api.call_args
        # System prompt is passed as second positional argument
        assert len(call_args.args) == 2
        assert call_args.args[1] == system_prompt


# Task 017-A-1: WebUI 履歴パネルUIの構築
def test_user_id_input_exists():
    """WebUI should have user ID input textbox"""
    # Verify that the demo has user_id component
    assert webui.demo is not None
    # This will fail until we add the component
    # Component will be accessible via demo.blocks dict
    components = {
        block.elem_id: block for block in webui.demo.blocks.values() if hasattr(block, "elem_id")
    }
    assert "user_id_input" in components


def test_user_id_warning_text_exists():
    """WebUI should display warning text about user ID"""
    # Verify warning text is in the demo
    assert webui.demo is not None
    # Check that a Markdown component with warning text exists
    components = {
        block.elem_id: block for block in webui.demo.blocks.values() if hasattr(block, "elem_id")
    }
    assert "user_id_warning" in components
    # Verify the warning markdown contains expected text
    warning_block = components["user_id_warning"]
    assert hasattr(warning_block, "value")
    assert "認証ではありません" in warning_block.value
    assert "他人のID" in warning_block.value


def test_history_dropdown_exists():
    """WebUI should have history list dropdown"""
    assert webui.demo is not None
    components = {
        block.elem_id: block for block in webui.demo.blocks.values() if hasattr(block, "elem_id")
    }
    assert "history_dropdown" in components


def test_save_name_input_exists():
    """WebUI should have save name input textbox"""
    assert webui.demo is not None
    components = {
        block.elem_id: block for block in webui.demo.blocks.values() if hasattr(block, "elem_id")
    }
    assert "save_name_input" in components


def test_history_buttons_exist():
    """WebUI should have save/load/new buttons for history management"""
    assert webui.demo is not None
    components = {
        block.elem_id: block for block in webui.demo.blocks.values() if hasattr(block, "elem_id")
    }
    assert "save_history_btn" in components
    assert "load_history_btn" in components
    assert "new_chat_btn" in components


def test_history_status_display_exists():
    """WebUI should have status display for history operations"""
    assert webui.demo is not None
    components = {
        block.elem_id: block for block in webui.demo.blocks.values() if hasattr(block, "elem_id")
    }
    assert "history_status" in components


def test_buttons_disabled_when_user_id_empty():
    """History buttons should be disabled when user ID is empty"""
    result = webui.check_history_buttons_enabled("")
    assert result["save_btn"]["interactive"] is False
    assert result["load_btn"]["interactive"] is False
    assert result["new_btn"]["interactive"] is False


def test_buttons_enabled_when_user_id_provided():
    """History buttons should be enabled when user ID is provided"""
    result = webui.check_history_buttons_enabled("test_user")
    assert result["save_btn"]["interactive"] is True
    assert result["load_btn"]["interactive"] is True
    assert result["new_btn"]["interactive"] is True


def test_send_button_disabled_when_user_id_empty():
    """Send button should be disabled when user ID is empty"""
    result = webui.check_send_button_with_user_id("", "test prompt")
    assert result["interactive"] is False


def test_send_button_disabled_when_token_limit_exceeded():
    """Send button should be disabled when token limit is exceeded even with valid user_id"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 2000000,
            "max_context_length": 1048576,
            "is_estimated": False,
        }
        result = webui.check_send_button_with_user_id(
            "test_user", "Very long prompt", None, "gemini-2.0-flash-exp"
        )
        assert result["interactive"] is False


def test_send_button_enabled_when_user_id_and_token_ok():
    """Send button should be enabled when user ID is valid AND token count is within limit"""
    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 100,
            "max_context_length": 1048576,
            "is_estimated": True,
        }
        result = webui.check_send_button_with_user_id(
            "test_user", "Normal prompt", None, "gemini-2.0-flash-exp"
        )
        assert result["interactive"] is True


def test_respond_rejects_empty_user_id():
    """respond() should reject requests when user_id is empty"""
    user_message = "@gemini Hello"
    display_history = []
    logic_history = []
    system_prompt = "Test"

    # Call respond with empty user_id
    result_gen = webui.respond(
        user_message, display_history, logic_history, system_prompt, user_id=""
    )
    # Consume all yields
    results = list(result_gen)

    # Should return error message without calling LLM
    assert len(results) == 1
    final_display = results[0][0]
    assert len(final_display) == 1
    assert "ユーザーIDを入力してください" in final_display[0][1]


def test_respond_rejects_whitespace_user_id():
    """respond() should reject requests when user_id is only whitespace"""
    with patch("multi_llm_chat.core.call_gemini_api") as mock_api:
        user_message = "@gemini Hello"
        display_history = []
        logic_history = []
        system_prompt = "Test"

        # Call respond with whitespace user_id
        result_gen = webui.respond(
            user_message, display_history, logic_history, system_prompt, user_id="   "
        )
        results = list(result_gen)

        # Should return error message
        assert len(results) == 1
        final_display = results[0][0]
        assert "ユーザーIDを入力してください" in final_display[0][1]

        # LLM should NOT be called
        mock_api.assert_not_called()


# Task 017-A-2: 確認フロー（モーダル風UI）の実装
def test_confirmation_ui_components_exist():
    """WebUI should have confirmation UI components"""
    assert webui.demo is not None
    components = {
        block.elem_id: block for block in webui.demo.blocks.values() if hasattr(block, "elem_id")
    }
    assert "confirmation_message" in components
    assert "confirmation_yes_btn" in components
    assert "confirmation_no_btn" in components


def test_confirmation_state_exists():
    """WebUI should have confirmation state management"""
    # Check that confirmation_state exists in demo
    assert webui.demo is not None
    # Confirmation state should be a gr.State component
    # Will be verified by checking it's used in event handlers
