"""Tests for the WebUI package, refactored for the new architecture."""

import asyncio
from unittest.mock import patch

from multi_llm_chat.webui.components import update_token_display
from multi_llm_chat.webui.handlers import (
    check_history_name_exists,
    get_history_list,
    has_unsaved_session,
    load_history_action,
    new_chat_action,
    save_history_action,
    validate_and_respond,
)
from multi_llm_chat.webui.state import WebUIState


async def consume_async_gen(gen):
    """Helper to consume an async generator and return all yielded items."""
    results = []
    async for item in gen:
        results.append(item)
    return results


# --- Tests for components.py ---
def test_token_count_display_updates():
    """components.update_token_display: should update when system prompt changes"""
    with patch("multi_llm_chat.webui.components.core.CHATGPT_MODEL", "gpt-4o"):
        with patch("multi_llm_chat.webui.components.core.get_token_info") as mock_token_info:
            mock_token_info.return_value = {
                "token_count": 50,
                "max_context_length": 1048576,
                "is_estimated": True,
            }
            result = update_token_display("Test system prompt")
            assert "50" in result
            assert "1048576" in result
            assert "estimated" in result.lower()
            mock_token_info.assert_called_once_with("Test system prompt", "gpt-4o", None)


def test_token_limit_warning_display():
    """components.update_token_display: should show warning when limit exceeded"""
    with patch("multi_llm_chat.webui.components.core.get_token_info") as mock_token_info:
        mock_token_info.return_value = {
            "token_count": 2000000,
            "max_context_length": 1048576,
            "is_estimated": False,
        }
        result = update_token_display("Very long prompt")
        assert "警告" in result or "warning" in result.lower()


# --- Tests for state.py (WebUIState) ---
class TestWebUIState:
    def test_buttons_disabled_when_user_id_empty(self):
        """WebUIState: All buttons should be disabled or limited when user_id is empty."""
        state = WebUIState(user_id="", has_history=False, is_streaming=False)
        buttons = state.get_button_states()
        assert buttons["send_button"]["interactive"] is False
        assert buttons["new_chat_btn"]["interactive"] is False
        assert buttons["reset_button"]["interactive"] is False
        assert buttons["save_history_btn"]["interactive"] is False
        # load_history_btn might be enabled to allow user to load a history, which sets the user_id
        assert buttons["load_history_btn"]["interactive"] is False

    def test_buttons_for_new_user_without_history(self):
        """WebUIState: Test button states for a new user without any saved history."""
        state = WebUIState(
            user_id="new_user", has_history=False, is_streaming=False, logic_history=[]
        )
        buttons = state.get_button_states()
        assert buttons["send_button"]["interactive"] is True
        assert buttons["new_chat_btn"]["interactive"] is True
        assert buttons["save_history_btn"]["interactive"] is False  # No conversation yet
        assert buttons["load_history_btn"]["interactive"] is False  # No saved history

    def test_save_button_enabled_with_conversation(self):
        """WebUIState: Save button should be enabled once a conversation starts."""
        state = WebUIState(
            user_id="test_user",
            has_history=False,
            is_streaming=False,
            logic_history=[{"role": "user", "content": "Hello"}],  # Conversation started
        )
        buttons = state.get_button_states()
        assert buttons["save_history_btn"]["interactive"] is True

    def test_buttons_for_user_with_saved_history(self):
        """WebUIState: Load button should be enabled if the user has saved history."""
        state = WebUIState(user_id="test_user", has_history=True, is_streaming=False)
        buttons = state.get_button_states()
        assert buttons["load_history_btn"]["interactive"] is True
        assert buttons["new_chat_btn"]["interactive"] is True  # Should always be available

    def test_send_button_disabled_when_token_limit_exceeded(self):
        """WebUIState: Send button should be disabled when token limit is exceeded."""
        with patch("multi_llm_chat.core.get_token_info") as mock_token_info:
            mock_token_info.return_value = {
                "token_count": 2000000,
                "max_context_length": 1048576,
            }
            state = WebUIState(
                user_id="test_user",
                has_history=True,
                is_streaming=False,
                system_prompt="A very long prompt",
            )
            buttons = state.get_button_states()
            assert buttons["send_button"]["interactive"] is False

    def test_all_buttons_disabled_while_streaming(self):
        """WebUIState: All interactive buttons should be disabled while streaming."""
        state = WebUIState(user_id="test_user", has_history=True, is_streaming=True)
        buttons = state.get_button_states()
        assert buttons["send_button"]["interactive"] is False
        assert buttons["new_chat_btn"]["interactive"] is False
        assert buttons["reset_button"]["interactive"] is False
        assert buttons["save_history_btn"]["interactive"] is False
        # load can be disabled as well, as it would interrupt the stream
        assert buttons["load_history_btn"]["interactive"] is False


# --- Tests for handlers.py ---
class TestWebUIHandlers:
    def test_validate_and_respond_rejects_empty_user_id(self):
        """handlers.validate_and_respond: should reject requests when user_id is empty"""
        result_gen = validate_and_respond(
            "Hi",
            display_history=[],
            logic_history=[],
            system_prompt="",
            user_id="",
            chat_service=None,
        )
        results = asyncio.run(consume_async_gen(result_gen))
        assert len(results) == 1
        final_display = results[0][0]
        assert "ユーザーIDを入力してください" in final_display[0][1]

    def test_validate_and_respond_delegates_to_respond(self):
        """handlers.validate_and_respond: should delegate to respond() when user_id is valid."""
        with patch("multi_llm_chat.webui.handlers.respond") as mock_respond:

            async def mock_respond_gen(*args, **kwargs):
                yield ("display", "display", "logic", "service")

            mock_respond.side_effect = mock_respond_gen
            result_gen = validate_and_respond("Hi", [], [], "", "test_user", None)
            asyncio.run(consume_async_gen(result_gen))  # Consume generator
            mock_respond.assert_called_once()

    def test_system_prompt_included_in_chat(self):
        """handlers.respond: should include system prompt when calling LLM via ChatService"""
        system_prompt = "You are a helpful assistant."
        with patch("multi_llm_chat.webui.handlers.ChatService") as MockChatService:
            mock_service_instance = MockChatService.return_value

            async def mock_process_message(*args, **kwargs):
                yield (["Hi"], ["Hi"], {"type": "text", "content": "Hi"})

            mock_service_instance.process_message.side_effect = mock_process_message

            # We test the wrapper `validate_and_respond` which calls the actual respond
            result_gen = validate_and_respond(
                "@gemini Hello", [], [], system_prompt, "test_user", None
            )
            asyncio.run(consume_async_gen(result_gen))  # consume generator

            # Check that the service was initialized and its state was set correctly
            assert mock_service_instance.system_prompt == system_prompt
            mock_service_instance.process_message.assert_called_once_with("@gemini Hello")

    def test_save_history_action_saves_to_file(self):
        """handlers.save_history_action: should save history to file using HistoryStore"""
        with patch("multi_llm_chat.webui.handlers.HistoryStore") as MockStore:
            mock_store_instance = MockStore.return_value
            mock_store_instance.list_histories.return_value = ["test_history"]

            status, choices = save_history_action("user", "test", [], "prompt")

            mock_store_instance.save_history.assert_called_once_with("user", "test", "prompt", [])
            assert "保存しました" in status
            assert "test_history" in choices

    def test_load_history_action_loads_from_file(self):
        """handlers.load_history_action: should load history from file using HistoryStore"""
        with patch("multi_llm_chat.webui.handlers.HistoryStore") as MockStore:
            mock_store_instance = MockStore.return_value
            mock_store_instance.load_history.return_value = {"system_prompt": "Test", "turns": []}

            _, _, sys_prompt, status = load_history_action("user", "test")

            mock_store_instance.load_history.assert_called_once_with("user", "test")
            assert sys_prompt == "Test"
            assert "読み込みました" in status

    def test_load_history_action_preserves_tool_logs(self):
        """handlers.load_history_action: should preserve tool execution logs in display"""
        with patch("multi_llm_chat.webui.handlers.HistoryStore") as MockStore:
            mock_store_instance = MockStore.return_value
            # Simulate saved history with tool interactions (matching agentic_loop format)
            mock_store_instance.load_history.return_value = {
                "system_prompt": "Test",
                "turns": [
                    {"role": "user", "content": [{"type": "text", "content": "Use a tool"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "content": {
                                    "name": "search",
                                    "arguments": {"query": "test"},
                                    "tool_call_id": "call_abc123",
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool_result",
                                "name": "search",
                                "content": "Found 5 results",
                                "tool_call_id": "call_abc123",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "content": "Here are the results"}],
                    },
                ],
            }

            display, logic, sys_prompt, status = load_history_action("user", "test")

            # Verify tool logs are in display history
            assert display is not None
            # Should have one turn with tool logs embedded in assistant response
            assert len(display) == 1
            user_msg, assistant_msg = display[0]
            assert "Use a tool" in user_msg
            # Tool logs should be formatted with emojis
            assert "🔧" in assistant_msg  # Tool call indicator
            assert "search" in assistant_msg
            assert "✅" in assistant_msg  # Tool result indicator
            assert "Found 5 results" in assistant_msg
            assert "Here are the results" in assistant_msg
            assert "読み込みました" in status

    def test_load_history_action_handles_file_not_found(self):
        """handlers.load_history_action: should handle FileNotFoundError"""
        with patch("multi_llm_chat.webui.handlers.HistoryStore") as MockStore:
            mock_store_instance = MockStore.return_value
            mock_store_instance.load_history.side_effect = FileNotFoundError

            res = load_history_action("user", "test")

            assert res == (None, None, None, "❌ 履歴 'test' が見つかりません")

    def test_check_history_name_exists_uses_historystore(self):
        """handlers.check_history_name_exists: should use HistoryStore.history_exists"""
        with patch("multi_llm_chat.webui.handlers.HistoryStore") as MockStore:
            mock_store_instance = MockStore.return_value
            mock_store_instance.history_exists.return_value = True

            result = check_history_name_exists("user", "name")

            mock_store_instance.history_exists.assert_called_once_with("user", "name")
            assert result is True

    def test_get_history_list_returns_choices(self):
        """handlers.get_history_list: should return list of saved histories"""
        with patch("multi_llm_chat.webui.handlers.HistoryStore") as MockStore:
            mock_store_instance = MockStore.return_value
            mock_store_instance.list_histories.return_value = ["h1", "h2"]

            result = get_history_list("user")

            mock_store_instance.list_histories.assert_called_once_with("user")
            assert result == ["h1", "h2"]

    def test_get_history_list_empty_on_empty_user_id(self):
        """handlers.get_history_list: should return empty list for empty user_id"""
        assert get_history_list("") == []
        assert get_history_list("   ") == []

    def test_has_unsaved_session(self):
        """handlers.has_unsaved_session: should return True if history has content"""
        assert has_unsaved_session([]) is False
        assert has_unsaved_session([{"role": "user", "content": "a"}]) is True

    def test_new_chat_action_resets_state(self):
        """handlers.new_chat_action: should return empty state tuple"""
        display, logic, sys_prompt, status = new_chat_action()
        assert display == []
        assert logic == []
        assert sys_prompt == ""
        assert "新しい会話を開始しました" in status

    def test_webui_displays_tool_calls(self):
        """WebUI displays tool calls with Markdown formatting."""
        import asyncio

        async def run_test():
            from multi_llm_chat.webui.handlers import format_tool_response

            # Test tool call formatting
            tool_call = {"name": "get_weather", "arguments": {"location": "Tokyo"}}
            tool_call_text = format_tool_response("tool_call", tool_call)

            assert "🔧 **Tool Call**" in tool_call_text
            assert "get_weather" in tool_call_text

            # Test tool result formatting
            tool_result = {"name": "get_weather", "content": "The temperature is 25°C"}
            tool_result_text = format_tool_response("tool_result", tool_result)

            assert "✅ **Result**" in tool_result_text
            assert "25°C" in tool_result_text

        asyncio.run(run_test())
