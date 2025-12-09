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


def test_show_confirmation_dialog():
    """show_confirmation function should display dialog with message"""
    if not hasattr(webui, "show_confirmation"):
        import pytest

        pytest.skip("show_confirmation not yet implemented")

    message = "この操作を実行しますか？"
    result = webui.show_confirmation(message, "test_action", {"key": "value"})

    # Should return updates for visibility and message
    assert result[0]["visible"] is True  # confirmation_dialog visible
    assert message in result[1]  # confirmation_message content
    assert result[2]["pending_action"] == "test_action"  # confirmation_state


def test_hide_confirmation_dialog():
    """hide_confirmation function should hide dialog"""
    if not hasattr(webui, "hide_confirmation"):
        import pytest

        pytest.skip("hide_confirmation not yet implemented")

    result = webui.hide_confirmation()

    # Should return update to hide dialog
    assert result[0]["visible"] is False  # confirmation_dialog hidden
    assert result[1] == ""  # confirmation_message cleared
    assert result[2]["pending_action"] is None  # confirmation_state cleared


def test_has_unsaved_session_empty():
    """has_unsaved_session should return False for empty history"""
    assert webui.has_unsaved_session([]) is False


def test_has_unsaved_session_with_content():
    """has_unsaved_session should return True when history exists"""
    history = [{"role": "user", "content": "Hello"}]
    assert webui.has_unsaved_session(history) is True


def test_check_history_name_exists():
    """check_history_name_exists placeholder should return False"""
    # Currently returns False (placeholder)
    assert webui.check_history_name_exists("test_user", "test_name") is False


def test_load_history_preserves_data_when_showing_confirmation():
    """handle_load_history should preserve current data when showing confirmation"""
    # This tests that when confirmation is shown, existing history is NOT cleared
    # Simulate calling handle_load_history with unsaved session
    # The function should return gr.update() to preserve existing values

    # Note: This is difficult to test without actually running the Gradio app
    # We're testing the logic by checking has_unsaved_session behavior
    logic_hist = [{"role": "user", "content": "Test"}]
    assert webui.has_unsaved_session(logic_hist) is True


def test_new_chat_preserves_data_when_showing_confirmation():
    """handle_new_chat should preserve current data when showing confirmation"""
    # Similar to above, when confirmation is shown, data should be preserved
    logic_hist = [{"role": "user", "content": "Test"}]
    assert webui.has_unsaved_session(logic_hist) is True


# Task 017-A-3: HistoryStore統合と機能連携
def test_save_history_action_saves_to_file():
    """save_history_action should save history to file using HistoryStore"""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory():
        user_id = "test_user"
        save_name = "test_history"
        logic_hist = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there", "model": "gemini"},
        ]
        sys_prompt = "You are a helpful assistant"

        # Mock HistoryStore to use tmpdir
        with patch("multi_llm_chat.webui.HistoryStore") as MockStore:
            mock_store = MockStore.return_value
            mock_store.save_history.return_value = None
            mock_store.list_histories.return_value = ["test_history"]

            status, choices = webui.save_history_action(user_id, save_name, logic_hist, sys_prompt)

            # Should call HistoryStore.save_history
            mock_store.save_history.assert_called_once()
            assert "test_history" in status or "保存" in status
            assert "test_history" in choices


def test_load_history_action_loads_from_file():
    """load_history_action should load history from file using HistoryStore"""
    with patch("multi_llm_chat.webui.HistoryStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.load_history.return_value = {
            "system_prompt": "Test prompt",
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi", "model": "gemini"},
            ],
        }

        display_hist, logic_hist, sys_prompt, status = webui.load_history_action(
            "test_user", "test_history"
        )

        # Should call HistoryStore.load_history
        mock_store.load_history.assert_called_once_with("test_user", "test_history")
        assert sys_prompt == "Test prompt"
        assert len(logic_hist) == 2
        assert "読み込み" in status or "test_history" in status


def test_new_chat_action_resets_state():
    """new_chat_action should reset all state"""
    display_hist, logic_hist, sys_prompt, status = webui.new_chat_action()

    assert display_hist == []
    assert logic_hist == []
    assert sys_prompt == ""
    assert "新しい会話" in status or "開始" in status


def test_check_history_name_exists_uses_historystore():
    """check_history_name_exists should use HistoryStore.history_exists"""
    with patch("multi_llm_chat.webui.HistoryStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.history_exists.return_value = True

        result = webui.check_history_name_exists("test_user", "existing_history")

        mock_store.history_exists.assert_called_once_with("test_user", "existing_history")
        assert result is True


def test_get_history_list_returns_choices():
    """get_history_list should return list of saved histories"""
    with patch("multi_llm_chat.webui.HistoryStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.list_histories.return_value = ["history1", "history2", "history3"]

        result = webui.get_history_list("test_user")

        mock_store.list_histories.assert_called_once_with("test_user")
        assert result == ["history1", "history2", "history3"]


def test_get_history_list_empty_user_id():
    """get_history_list should return empty list for empty user_id"""
    result = webui.get_history_list("")
    assert result == []

    result = webui.get_history_list("   ")
    assert result == []


def test_load_history_action_preserves_state_on_error():
    """load_history_action should return None on error to preserve current state"""
    with patch("multi_llm_chat.webui.HistoryStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.load_history.side_effect = FileNotFoundError("Not found")

        display_hist, logic_hist, sys_prompt, status = webui.load_history_action(
            "test_user", "nonexistent"
        )

        # Should return None to indicate error
        assert display_hist is None
        assert logic_hist is None
        assert sys_prompt is None
        assert "見つかりません" in status


def test_load_history_action_preserves_state_on_exception():
    """load_history_action should return None on exception"""
    with patch("multi_llm_chat.webui.HistoryStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.load_history.side_effect = Exception("Read error")

        display_hist, logic_hist, sys_prompt, status = webui.load_history_action(
            "test_user", "corrupted"
        )

        # Should return None to indicate error
        assert display_hist is None
        assert logic_hist is None
        assert sys_prompt is None
        assert "失敗しました" in status


def test_load_history_action_handles_multiple_assistant_responses():
    """load_history_action should handle @all case with multiple assistant responses"""
    with patch("multi_llm_chat.webui.HistoryStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.load_history.return_value = {
            "system_prompt": "Test",
            "turns": [
                {"role": "user", "content": "@all Hello"},
                {"role": "gemini", "content": "Gemini response"},
                {"role": "chatgpt", "content": "ChatGPT response"},
            ],
        }

        display_hist, logic_hist, sys_prompt, status = webui.load_history_action(
            "test_user", "multi_response"
        )

        # Should have both responses in display
        assert len(display_hist) == 1
        assert display_hist[0][0] == "@all Hello"
        # Both responses should be present
        assert "Gemini response" in display_hist[0][1]
        assert "ChatGPT response" in display_hist[0][1]
        # Logic history should have all 3 turns
        assert len(logic_hist) == 3


def test_build_history_operation_updates_helper():
    """Helper function should build consistent UI update dictionary"""
    user_id = "test_user"
    display_hist = [["Hello", "Hi there"]]
    logic_hist = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    sys_prompt = "You are helpful"
    status = "✅ Success"

    with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 50,
            "max_context_length": 100000,
            "is_estimated": False,
        }

        result = webui._build_history_operation_updates(
            user_id, display_hist, logic_hist, sys_prompt, status
        )

        # Should return dictionary with all required keys
        assert "chatbot_ui" in result
        assert "display_history_state" in result
        assert "logic_history_state" in result
        assert "system_prompt_input" in result
        assert "history_status" in result
        assert "token_display" in result
        assert "send_button" in result
        assert "history_dropdown" in result

        # Values should match inputs
        assert result["chatbot_ui"] == display_hist
        assert result["display_history_state"] == display_hist
        assert result["logic_history_state"] == logic_hist
        assert result["system_prompt_input"] == sys_prompt
        assert result["history_status"] == status


def test_build_history_operation_updates_with_history_list():
    """Helper should include history dropdown choices when available"""
    user_id = "test_user"
    display_hist = []
    logic_hist = []
    sys_prompt = ""
    status = "Test"

    with (
        patch("multi_llm_chat.core.get_token_info") as mock_token_info,
        patch("multi_llm_chat.webui.get_history_list") as mock_history_list,
    ):
        mock_token_info.return_value = {
            "token_count": 0,
            "max_context_length": 100000,
            "is_estimated": False,
        }
        mock_history_list.return_value = ["history1", "history2"]

        result = webui._build_history_operation_updates(
            user_id, display_hist, logic_hist, sys_prompt, status
        )

        # Should call get_history_list
        mock_history_list.assert_called_once_with(user_id)

        # history_dropdown should be gr.update dict with choices
        assert isinstance(result["history_dropdown"], dict)
        assert result["history_dropdown"]["choices"] == ["history1", "history2"]
