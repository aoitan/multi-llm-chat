"""Tests for ChatService - business logic layer for chat operations"""

import unittest
from unittest.mock import MagicMock, patch

from multi_llm_chat.chat_logic import ChatService, parse_mention


class TestChatServiceBasics(unittest.TestCase):
    """Test basic ChatService initialization and state management"""

    def test_create_service_with_empty_history(self):
        """ChatService should initialize with empty histories"""
        service = ChatService()

        assert service.display_history == []
        assert service.logic_history == []
        assert service.system_prompt == ""

    def test_create_service_with_existing_data(self):
        """ChatService should accept initial state"""
        display_hist = [["user msg", "assistant msg"]]
        logic_hist = [
            {"role": "user", "content": "user msg"},
            {"role": "assistant", "content": "assistant msg"},
        ]
        sys_prompt = "Test prompt"

        service = ChatService(
            display_history=display_hist,
            logic_history=logic_hist,
            system_prompt=sys_prompt,
        )

        assert service.display_history == display_hist
        assert service.logic_history == logic_hist
        assert service.system_prompt == sys_prompt


class TestChatServiceMessageParsing(unittest.TestCase):
    """Test mention parsing logic"""

    def test_parse_mention_gemini(self):
        """Should detect @gemini mention"""
        mention = parse_mention("@gemini tell me about Python")
        assert mention == "gemini"

    def test_parse_mention_chatgpt(self):
        """Should detect @chatgpt mention"""
        mention = parse_mention("@chatgpt explain async")
        assert mention == "chatgpt"

    def test_parse_mention_all(self):
        """Should detect @all mention"""
        mention = parse_mention("@all compare these two")
        assert mention == "all"

    def test_parse_mention_none(self):
        """Should return None for messages without mentions"""
        mention = parse_mention("regular message")
        assert mention is None

    def test_parse_mention_ignores_whitespace(self):
        """Should handle leading/trailing whitespace"""
        mention = parse_mention("  @gemini  ")
        assert mention == "gemini"


class TestChatServiceProcessMessage(unittest.TestCase):
    """Test main message processing logic"""

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_process_message_gemini(self, mock_create_provider):
        """Should call Gemini API for @gemini mention"""
        # Setup mock provider
        mock_provider = MagicMock()
        mock_provider.call_api.return_value = iter(
            [{"type": "text", "content": "Test "}, {"type": "text", "content": "response"}]
        )
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = list(service.process_message("@gemini hello"))

        # Should have yielded at least once
        assert len(results) > 0

        # Final state should include user message and response
        final_display, final_logic = results[-1]
        assert len(final_logic) == 2  # user + assistant
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[1]["content"] == [{"type": "text", "content": "Test response"}]

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_process_message_chatgpt(self, mock_create_provider):
        """Should call ChatGPT API for @chatgpt mention"""
        # Setup mock provider
        mock_provider = MagicMock()
        mock_provider.call_api.return_value = iter(
            [{"type": "text", "content": "Hello "}, {"type": "text", "content": "world"}]
        )
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = list(service.process_message("@chatgpt hi"))

        final_display, final_logic = results[-1]
        assert len(final_logic) == 2
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "chatgpt"
        assert final_logic[1]["content"] == [{"type": "text", "content": "Hello world"}]

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_process_message_chatgpt_logs_warning_when_tools_provided(self, mock_create_provider):
        """ChatGPT should log warning when tools are passed (tools=None is sent to API)."""
        mock_provider = MagicMock()
        # Return successful response since tools=None is passed
        mock_provider.call_api.return_value = iter([{"type": "text", "content": "Hello"}])
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        # Pass tools to process_message
        with self.assertLogs("multi_llm_chat.chat_logic", level="WARNING") as log_context:
            list(service.process_message("@chatgpt hi", tools=[{"name": "search"}]))

        # Check that warning was logged
        assert any("ChatGPT does not support tools" in message for message in log_context.output), (
            "Warning should be logged when tools are provided"
        )

        # Verify that tools=None was passed to API
        mock_provider.call_api.assert_called_once()
        call_kwargs = mock_provider.call_api.call_args[1]
        assert call_kwargs.get("tools") is None

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_process_message_gemini_tool_call(self, mock_create_provider):
        """Gemini tool_call chunks should be stored in structured content."""
        mock_provider = MagicMock()
        mock_provider.call_api.return_value = iter(
            [
                {
                    "type": "tool_call",
                    "content": {"name": "search", "arguments": {"query": "hi"}},
                }
            ]
        )
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = list(service.process_message("@gemini hi"))

        final_display, final_logic = results[-1]
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[1]["content"] == [
            {
                "type": "tool_call",
                "content": {"name": "search", "arguments": {"query": "hi"}},
            }
        ]
        assert "[Tool Call: search]" in final_display[-1][1]

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_process_message_gemini_preserves_stream_order(self, mock_create_provider):
        """Gemini text/tool_callのストリーム順序を履歴に反映できること"""
        mock_provider = MagicMock()
        mock_provider.call_api.return_value = iter(
            [
                {"type": "text", "content": "Before "},
                {"type": "tool_call", "content": {"name": "search", "arguments": {}}},
                {"type": "text", "content": "after"},
            ]
        )
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = list(service.process_message("@gemini check"))

        _final_display, final_logic = results[-1]
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[1]["content"] == [
            {"type": "text", "content": "Before "},
            {"type": "tool_call", "content": {"name": "search", "arguments": {}}},
            {"type": "text", "content": "after"},
        ]

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_process_message_all(self, mock_create_provider):
        """Should call both APIs for @all mention"""
        # Setup mock providers for both calls
        mock_gemini_provider = MagicMock()
        mock_gemini_provider.call_api.return_value = iter(
            [{"type": "text", "content": "Gemini response"}]
        )

        mock_chatgpt_provider = MagicMock()
        mock_chatgpt_provider.call_api.return_value = iter(
            [{"type": "text", "content": "ChatGPT response"}]
        )

        # Return different providers for gemini and chatgpt
        mock_create_provider.side_effect = [mock_gemini_provider, mock_chatgpt_provider]

        service = ChatService()
        results = list(service.process_message("@all compare"))

        final_display, final_logic = results[-1]
        # Should have user message + 2 responses
        assert len(final_logic) == 3
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[2]["role"] == "chatgpt"

    def test_process_message_no_mention_as_memo(self):
        """Messages without mention should be added to history as memo (no LLM call)"""
        service = ChatService()

        results = list(service.process_message("This is a memo"))

        # Should yield once (user message added to history)
        assert len(results) == 1
        final_display, final_logic = results[0]

        # User message should be in history (structured format)
        assert len(final_logic) == 1
        assert final_logic[0]["role"] == "user"
        assert final_logic[0]["content"] == [{"type": "text", "content": "This is a memo"}]

        # Display should show user message with no response
        assert len(final_display) == 1
        assert final_display[0][0] == "This is a memo"
        assert final_display[0][1] is None

    def test_append_tool_results_adds_tool_entry(self):
        """tool_resultを履歴に追加できること"""
        service = ChatService()
        tool_results = [
            {"name": "get_weather", "content": '{"temperature": "25C"}', "tool_call_id": "t1"}
        ]

        service.append_tool_results(tool_results)

        assert len(service.logic_history) == 1
        assert service.logic_history[0]["role"] == "tool"
        assert service.logic_history[0]["content"] == [
            {
                "type": "tool_result",
                "name": "get_weather",
                "content": '{"temperature": "25C"}',
                "tool_call_id": "t1",
            }
        ]

    def test_append_tool_results_logs_invalid_input(self):
        """不正な型のツール結果が渡された場合に警告ログを出力すること"""
        service = ChatService()

        # Mix valid and invalid tool results
        tool_results = [
            {"name": "valid", "content": "ok"},
            "invalid_string",  # Should be logged and skipped
            123,  # Should be logged and skipped
            {"name": "valid2", "content": "ok2"},
        ]

        with self.assertLogs("multi_llm_chat.chat_logic", level="WARNING") as log_context:
            service.append_tool_results(tool_results)

        # Check that warnings were logged for invalid entries
        assert any("Invalid tool result type: str" in message for message in log_context.output)
        assert any("Invalid tool result type: int" in message for message in log_context.output)

        # Check that only valid results were added
        assert len(service.logic_history) == 1
        assert len(service.logic_history[0]["content"]) == 2
        assert service.logic_history[0]["content"][0]["name"] == "valid"
        assert service.logic_history[0]["content"][1]["name"] == "valid2"


class TestChatServiceHistorySnapshot(unittest.TestCase):
    """Test history snapshot logic for @all"""

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_all_uses_same_history_snapshot(self, mock_create_provider):
        """@all should use identical history for both LLMs"""
        captured_histories = []

        def create_mock_provider(provider_name):
            mock_provider = MagicMock()

            def capture_stream_gemini(history, system_prompt=None, tools=None):
                captured_histories.append(("gemini", [h.copy() for h in history]))
                return iter([{"type": "text", "content": "Gemini"}])

            def capture_stream_chatgpt(history, system_prompt=None, tools=None):
                captured_histories.append(("chatgpt", [h.copy() for h in history]))
                return iter([{"type": "text", "content": "ChatGPT"}])

            if provider_name == "gemini":
                mock_provider.call_api.side_effect = capture_stream_gemini
            else:  # chatgpt
                mock_provider.call_api.side_effect = capture_stream_chatgpt
            return mock_provider

        # Return different providers for each call
        mock_create_provider.side_effect = [
            create_mock_provider("gemini"),
            create_mock_provider("chatgpt"),
        ]

        service = ChatService()
        list(service.process_message("@all test"))

        # Both should have been called
        assert len(captured_histories) == 2

        # Sort by provider name to ensure consistent order
        captured_histories.sort(key=lambda x: x[0])

        gemini_hist = captured_histories[0][1]
        chatgpt_hist = captured_histories[1][1]

        # Both should be identical and contain only user message
        assert gemini_hist == chatgpt_hist
        assert len(gemini_hist) == 1
        assert gemini_hist[0]["role"] == "user"


class TestChatServiceSystemPrompt(unittest.TestCase):
    """Test system prompt handling"""

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_system_prompt_passed_to_api(self, mock_create_provider):
        """System prompt should be passed to LLM API"""
        mock_provider = MagicMock()
        mock_provider.call_api.return_value = iter([{"type": "text", "content": "Response"}])
        mock_create_provider.return_value = mock_provider

        service = ChatService(system_prompt="You are a helpful assistant")
        list(service.process_message("@gemini hello"))

        # Check that system prompt was passed (2nd positional argument to call_api)
        mock_provider.call_api.assert_called_once()
        call_args = mock_provider.call_api.call_args
        assert call_args[0][1] == "You are a helpful assistant"

    def test_update_system_prompt(self):
        """Should allow updating system prompt"""
        service = ChatService()
        service.set_system_prompt("New prompt")

        assert service.system_prompt == "New prompt"


class TestChatServiceErrorHandling(unittest.TestCase):
    """Test error handling for LLM API failures"""

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_network_error_handling(self, mock_create_provider):
        """Network errors should be caught and added to history as error message"""
        mock_provider = MagicMock()
        mock_provider.call_api.side_effect = ConnectionError("Network error")
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = list(service.process_message("@gemini hello"))

        # Should yield error message
        assert len(results) > 0
        final_display, final_logic = results[-1]

        # Error message should be in display history
        assert len(final_display) == 1
        assert "hello" in final_display[0][0]
        assert "[System: Gemini APIエラー" in final_display[0][1]
        assert "Network error" in final_display[0][1]

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_api_error_handling(self, mock_create_provider):
        """API errors should be caught and added to history as error message"""
        mock_provider = MagicMock()
        mock_provider.call_api.side_effect = ValueError("API key missing")
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = list(service.process_message("@chatgpt test"))

        # Should yield error message
        assert len(results) > 0
        final_display, final_logic = results[-1]

        # Error message should be in display history
        assert "[System: エラー" in final_display[0][1]
        assert "API key missing" in final_display[0][1]

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_gemini_empty_response_records_error(self, mock_create_provider):
        """空応答時にlogic_historyにエラーが記録されること

        (Issue #79: ツール呼び出し実装時に修正)
        """
        mock_provider = MagicMock()
        # ツール呼び出しなし、テキストなしのケース（空のストリーム）
        mock_provider.call_api.return_value = iter([])
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = list(service.process_message("@gemini hello"))

        final_display, final_logic = results[-1]

        # display_historyにエラーメッセージが表示されること
        assert len(final_display) == 1
        assert "応答がありませんでした" in final_display[0][1]

        # logic_historyにもgeminiのエラーメッセージが記録されること (修正後の動作)
        gemini_entries = [e for e in final_logic if e.get("role") == "gemini"]
        assert len(gemini_entries) == 1
        # 空の応答の場合、明示的なエラーメッセージがcontentに含まれる
        assert gemini_entries[0]["content"] == [
            {"type": "text", "content": "[System: Geminiからの応答がありませんでした]"}
        ]

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_all_handles_partial_failure(self, mock_create_provider):
        """@all should handle when one LLM succeeds and one fails"""
        # Gemini succeeds, ChatGPT fails
        mock_gemini = MagicMock()
        mock_gemini.call_api.return_value = iter([{"type": "text", "content": "Gemini response"}])

        mock_chatgpt = MagicMock()
        mock_chatgpt.call_api.side_effect = RuntimeError("ChatGPT API error")

        mock_create_provider.side_effect = [mock_gemini, mock_chatgpt]

        service = ChatService()
        results = list(service.process_message("@all hello"))

        # Should get results
        assert len(results) > 0
        final_display, final_logic = results[-1]

        # Should have Gemini success and ChatGPT error
        assert len(final_display) >= 1
        # Check that error message is present (with actual error format)
        error_found = any(
            "[System:" in msg[1] and "エラー" in msg[1] for msg in final_display if msg[1]
        )
        assert error_found


class TestChatServiceEmptyResponseHandling(unittest.TestCase):
    """Test empty response handling (Issue #79 Review Fix)"""

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_empty_gemini_response_with_no_streaming_content(self, mock_create_provider):
        """Completely empty response should show error message."""
        service = ChatService()

        # Mock provider that returns empty content
        mock_provider = MagicMock()
        mock_provider.call_api.return_value = iter([])  # No chunks at all
        mock_create_provider.return_value = mock_provider

        results = list(service.process_message("@gemini test"))

        # Should have added error message to display_history
        final_display, final_logic = results[-1]
        assert len(final_display) == 1
        assert "[System: Geminiからの応答がありませんでした]" in final_display[0][1]

        # Logic history should also have error message
        assert len(final_logic) == 2  # user + gemini
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[1]["content"][0]["type"] == "text"
        assert "応答がありませんでした" in final_logic[1]["content"][0]["content"]

    @patch("multi_llm_chat.chat_logic.create_provider")
    def test_empty_gemini_response_with_partial_streaming_content(self, mock_create_provider):
        """Streaming with partial content should NOT show error message."""
        service = ChatService()

        # Mock provider that returns some text first, then ends
        mock_provider = MagicMock()
        mock_provider.call_api.return_value = iter(
            [
                {"type": "text", "content": "部分的な"},
                {"type": "text", "content": "応答"},
            ]
        )
        mock_create_provider.return_value = mock_provider

        results = list(service.process_message("@gemini test"))

        # Should NOT have error message (streaming succeeded partially)
        final_display, final_logic = results[-1]
        assert len(final_display) == 1
        # Should only have the gemini label + streamed text
        assert "部分的な応答" in final_display[0][1]
        assert "[System: Geminiからの応答がありませんでした]" not in final_display[0][1]

        # Logic history should have accumulated text (concatenated in one entry)
        assert len(final_logic) == 2
        assert final_logic[1]["content"][0]["content"] == "部分的な応答"


if __name__ == "__main__":
    unittest.main()
