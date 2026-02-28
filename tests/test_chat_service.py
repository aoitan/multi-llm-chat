"""Tests for ChatService - business logic layer for chat operations"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_llm_chat.chat_service import AUTOSAVE_FAILURE_WARNING, ChatService, parse_mention
from multi_llm_chat.core_modules.agentic_loop import AgenticLoopResult


async def consume_async_gen(gen):
    """Helper to consume an async generator and return all yielded items."""
    results = []
    async for item in gen:
        results.append(item)
    return results


class TestChatServiceBasics(unittest.IsolatedAsyncioTestCase):  # Issue #119: async test support
    """Test basic ChatService initialization and state management"""

    @pytest.mark.asyncio
    async def test_create_service_with_empty_history(self):
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


class TestChatServiceMessageParsing(
    unittest.IsolatedAsyncioTestCase
):  # Issue #119: async test support
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

    @pytest.mark.asyncio
    async def test_parse_mention_ignores_whitespace(self):
        """Should handle leading/trailing whitespace"""
        mention = parse_mention("  @gemini  ")
        assert mention == "gemini"


class TestChatServiceProcessMessage:
    """Test main message processing logic"""

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_process_message_gemini(self, mock_create_provider):
        """Should call Gemini API for @gemini mention"""
        # Setup mock provider
        mock_provider = MagicMock()
        mock_provider.name = "gemini"

        async def mock_call_api(*args, **kwargs):
            yield {"type": "text", "content": "Test "}
            yield {"type": "text", "content": "response"}

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = await consume_async_gen(service.process_message("@gemini hello"))

        # Should have yielded at least once
        assert len(results) > 0

        # Final state should include user message and response
        final_display, final_logic, _chunk = results[-1]
        assert len(final_logic) == 2  # user + assistant
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[1]["content"] == [{"type": "text", "content": "Test response"}]

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_process_message_chatgpt(self, mock_create_provider):
        """Should call ChatGPT API for @chatgpt mention"""
        # Setup mock provider
        mock_provider = MagicMock()
        mock_provider.name = "chatgpt"

        async def mock_call_api(*args, **kwargs):
            yield {"type": "text", "content": "Hello "}
            yield {"type": "text", "content": "world"}

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = await consume_async_gen(service.process_message("@chatgpt hi"))

        final_display, final_logic, _chunk = results[-1]
        assert len(final_logic) == 2
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "chatgpt"
        assert final_logic[1]["content"] == [{"type": "text", "content": "Hello world"}]

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_process_message_chatgpt_supports_tools(self, mock_create_provider):
        """ChatGPT should support tools and pass them to API."""
        mock_provider = MagicMock()
        mock_provider.name = "chatgpt"

        # Return successful response
        async def mock_call_api(*args, **kwargs):
            yield {"type": "text", "content": "Hello"}

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        tools = [{"name": "search"}]
        await consume_async_gen(service.process_message("@chatgpt hi", tools=tools))

        # Verify that tools were passed to API as keyword argument
        mock_provider.call_api.assert_called_once()
        call_kwargs = mock_provider.call_api.call_args[1]
        assert "tools" in call_kwargs
        # Verify that the user-provided tool is included
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search" in tool_names
        # MCP tools may also be included if MCP is enabled

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_process_message_gemini_tool_call(self, mock_create_provider):
        """Gemini tool_call chunks should be stored in structured content."""
        mock_provider = MagicMock()
        mock_provider.name = "gemini"

        async def mock_call_api(*args, **kwargs):
            yield {
                "type": "tool_call",
                "content": {"name": "search", "arguments": {"query": "hi"}},
            }
            # After tool_call, LLM is called again in execute_with_tools.
            # We need to yield something else or stop.
            yield {"type": "text", "content": "Done"}

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        mock_mcp = MagicMock()
        mock_mcp.list_tools = AsyncMock(return_value=[{"name": "search"}])
        mock_mcp.call_tool = AsyncMock(
            return_value={"content": [{"type": "text", "text": "result"}], "isError": False}
        )

        service = ChatService(mcp_client=mock_mcp)
        results = await consume_async_gen(service.process_message("@gemini hi"))

        final_display, final_logic, _chunk = results[-1]
        assert final_logic[1]["role"] == "gemini"
        # First entry: text (Done) + tool_call
        # execute_with_tools appends thought_text before tool_calls
        assert final_logic[1]["content"][0]["type"] == "text"
        assert final_logic[1]["content"][1]["type"] == "tool_call"
        assert final_logic[1]["content"][1]["name"] == "search"
        # Second entry: tool_result
        assert final_logic[2]["role"] == "tool"
        assert final_logic[2]["content"][0]["type"] == "tool_result"
        assert final_logic[2]["content"][0]["name"] == "search"

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_process_message_gemini_preserves_stream_order(self, mock_create_provider):
        """Gemini text/tool_callのストリーム順序を履歴に反映できること"""
        mock_provider = MagicMock()
        mock_provider.name = "gemini"

        async def mock_call_api(*args, **kwargs):
            yield {"type": "text", "content": "Before "}
            yield {"type": "tool_call", "content": {"name": "search", "arguments": {}}}

            # Stop here to avoid infinite loop or needing more yields
            # Actually execute_with_tools will call call_api again if tool_call was returned.
            # To avoid the second call returning the same thing, we can use a stateful mock.
            async def next_call(*a, **kw):
                yield {"type": "text", "content": "after"}

            mock_provider.call_api.side_effect = next_call

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        mock_mcp = MagicMock()
        mock_mcp.list_tools = AsyncMock(return_value=[{"name": "search"}])
        mock_mcp.call_tool = AsyncMock(
            return_value={"content": [{"type": "text", "text": "result"}], "isError": False}
        )

        service = ChatService(mcp_client=mock_mcp)
        results = await consume_async_gen(service.process_message("@gemini check"))

        _final_display, final_logic, _chunk = results[-1]
        assert final_logic[1]["role"] == "gemini"
        # First entry: text (Before) + tool_call
        assert final_logic[1]["content"][0] == {"type": "text", "content": "Before "}
        assert final_logic[1]["content"][1]["type"] == "tool_call"
        assert final_logic[1]["content"][1]["name"] == "search"
        # Second entry (index 2): tool_result
        assert final_logic[2]["role"] == "tool"
        assert final_logic[2]["content"][0]["type"] == "tool_result"
        assert final_logic[2]["content"][0]["name"] == "search"
        # Third entry (index 3): text (after)
        assert final_logic[3]["role"] == "gemini"
        assert final_logic[3]["content"][0] == {"type": "text", "content": "after"}

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_process_message_all(self, mock_create_provider):
        """Should call both APIs for @all mention"""
        # Setup mock providers for both calls
        mock_gemini_provider = MagicMock()
        mock_gemini_provider.name = "gemini"

        async def mock_call_api_gemini(*args, **kwargs):
            yield {"type": "text", "content": "Gemini response"}

        mock_gemini_provider.call_api.side_effect = mock_call_api_gemini

        mock_chatgpt_provider = MagicMock()
        mock_chatgpt_provider.name = "chatgpt"

        async def mock_call_api_chatgpt(*args, **kwargs):
            yield {"type": "text", "content": "ChatGPT response"}

        mock_chatgpt_provider.call_api.side_effect = mock_call_api_chatgpt

        # Return different providers for gemini and chatgpt
        mock_create_provider.side_effect = [mock_gemini_provider, mock_chatgpt_provider]

        service = ChatService()
        results = await consume_async_gen(service.process_message("@all compare"))

        final_display, final_logic, _chunk = results[-1]
        # Should have user message + 2 responses
        assert len(final_logic) == 3
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[2]["role"] == "chatgpt"

    @pytest.mark.asyncio
    async def test_process_message_no_mention_as_memo(self):
        """Messages without mention should be added to history as memo (no LLM call)"""
        service = ChatService()

        results = await consume_async_gen(service.process_message("This is a memo"))

        # Should yield once (user message added to history)
        assert len(results) == 1
        final_display, final_logic, _chunk = results[0]

        # User message should be in history (structured format)
        assert len(final_logic) == 1
        assert final_logic[0]["role"] == "user"
        assert final_logic[0]["content"] == [{"type": "text", "content": "This is a memo"}]

        # Display should show user message with no response
        assert len(final_display) == 1
        assert final_display[0]["content"] == "This is a memo"

    @pytest.mark.asyncio
    async def test_append_tool_results_adds_tool_entry(self):
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

    def test_append_tool_results_logs_invalid_input(self, caplog):
        """不正な型のツール結果が渡された場合に警告ログを出力すること"""
        import logging

        service = ChatService()

        # Mix valid and invalid tool results
        tool_results = [
            {"name": "valid", "content": "ok"},
            "invalid_string",  # Should be logged and skipped
            123,  # Should be logged and skipped
            {"name": "valid2", "content": "ok2"},
        ]

        with caplog.at_level(logging.WARNING, logger="multi_llm_chat.chat_logic"):
            service.append_tool_results(tool_results)

        # Check that warnings were logged for invalid entries
        log_lines = caplog.text.splitlines()
        assert any("Invalid tool result type: str" in message for message in log_lines)
        assert any("Invalid tool result type: int" in message for message in log_lines)

        # Check that only valid results were added
        assert len(service.logic_history) == 1
        assert len(service.logic_history[0]["content"]) == 2
        assert service.logic_history[0]["content"][0]["name"] == "valid"
        assert service.logic_history[0]["content"][1]["name"] == "valid2"


class TestChatServiceHistorySnapshot(
    unittest.IsolatedAsyncioTestCase
):  # Issue #119: async test support
    """Test history snapshot logic for @all"""

    @pytest.mark.asyncio
    @patch("multi_llm_chat.chat_service.create_provider")
    async def test_all_uses_same_history_snapshot(self, mock_create_provider):
        """@all should use identical history for both LLMs"""
        captured_histories = []

        def create_mock_provider(provider_name):
            mock_provider = MagicMock()
            mock_provider.name = provider_name

            async def capture_stream_gemini(history, system_prompt=None, tools=None):
                captured_histories.append(("gemini", [h.copy() for h in history]))
                yield {"type": "text", "content": "Gemini"}

            async def capture_stream_chatgpt(history, system_prompt=None, tools=None):
                captured_histories.append(("chatgpt", [h.copy() for h in history]))
                yield {"type": "text", "content": "ChatGPT"}

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
        await consume_async_gen(service.process_message("@all test"))

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


class TestChatServiceSystemPrompt(
    unittest.IsolatedAsyncioTestCase
):  # Issue #119: async test support
    """Test system prompt handling"""

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_api(self, mock_create_provider):
        """System prompt should be passed to LLM API"""
        mock_provider = MagicMock()
        mock_provider.name = "gemini"

        async def mock_call_api(*args, **kwargs):
            yield {"type": "text", "content": "Response"}

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        service = ChatService(system_prompt="You are a helpful assistant")
        await consume_async_gen(service.process_message("@gemini hello"))

        # Check that system prompt was passed (2nd positional argument to call_api)
        mock_provider.call_api.assert_called_once()
        call_args = mock_provider.call_api.call_args
        assert call_args[0][1] == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_update_system_prompt(self):
        """Should allow updating system prompt"""
        service = ChatService()
        service.set_system_prompt("New prompt")

        assert service.system_prompt == "New prompt"


class TestChatServiceErrorHandling(
    unittest.IsolatedAsyncioTestCase
):  # Issue #119: async test support
    """Test error handling for LLM API failures"""

    @pytest.mark.asyncio
    @patch("multi_llm_chat.chat_service.create_provider")
    async def test_network_error_handling(self, mock_create_provider):
        """Network errors should be caught and added to history as error message"""
        mock_provider = MagicMock()
        mock_provider.name = "gemini"

        async def mock_call_api(*args, **kwargs):
            if False:
                yield {}
            raise ConnectionError("Network error")

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = await consume_async_gen(service.process_message("@gemini hello"))

        # Should yield error message
        assert len(results) > 0
        final_display, final_logic, _chunk = results[-1]

        # Error message should be in display history
        assert len(final_display) == 2
        assert "hello" in final_display[0]["content"]
        assert "[System: Gemini APIエラー" in final_display[1]["content"]
        assert "Network error" in final_display[1]["content"]

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_create_provider):
        """API errors should be caught and added to history as error message"""
        mock_provider = MagicMock()
        mock_provider.name = "chatgpt"

        async def mock_call_api(*args, **kwargs):
            if False:
                yield {}
            raise ValueError("API key missing")

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = await consume_async_gen(service.process_message("@chatgpt test"))

        # Should yield error message
        assert len(results) > 0
        final_display, final_logic, _chunk = results[-1]

        # Error message should be in display history
        assert "[System: エラー" in final_display[1]["content"]
        assert "API key missing" in final_display[1]["content"]

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_gemini_empty_response_records_error(self, mock_create_provider):
        """空応答時にlogic_historyにエラーが記録されること

        (Issue #79: ツール呼び出し実装時に修正)
        """
        mock_provider = MagicMock()
        mock_provider.name = "gemini"

        # ツール呼び出しなし、テキストなしのケース（空のストリーム）
        async def mock_call_api(*args, **kwargs):
            if False:
                yield {}

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        service = ChatService()
        results = await consume_async_gen(service.process_message("@gemini hello"))

        final_display, final_logic, _chunk = results[-1]

        # display_historyにエラーメッセージが表示されること
        assert len(final_display) == 2
        assert "応答がありませんでした" in final_display[1]["content"]

        # logic_historyにもgeminiのエラーメッセージが記録されること (修正後の動作)
        gemini_entries = [e for e in final_logic if e.get("role") == "gemini"]
        assert len(gemini_entries) == 1
        # 空の応答の場合、明示的なエラーメッセージがcontentに含まれる
        assert gemini_entries[0]["content"] == [
            {"type": "text", "content": "[System: Geminiからの応答がありませんでした]"}
        ]

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_all_handles_partial_failure(self, mock_create_provider):
        """@all should handle when one LLM succeeds and one fails"""
        # Gemini succeeds, ChatGPT fails
        mock_gemini = MagicMock()
        mock_gemini.name = "gemini"

        async def mock_call_api_gemini(*args, **kwargs):
            yield {"type": "text", "content": "Gemini response"}

        mock_gemini.call_api.side_effect = mock_call_api_gemini

        mock_chatgpt = MagicMock()
        mock_chatgpt.name = "chatgpt"

        async def mock_call_api_chatgpt(*args, **kwargs):
            if False:
                yield {}
            raise RuntimeError("ChatGPT API error")

        mock_chatgpt.call_api.side_effect = mock_call_api_chatgpt

        mock_create_provider.side_effect = [mock_gemini, mock_chatgpt]

        service = ChatService()
        results = await consume_async_gen(service.process_message("@all hello"))

        # Should get results
        assert len(results) > 0
        final_display, final_logic, _chunk = results[-1]

        # Should have Gemini success and ChatGPT error
        assert len(final_display) >= 1
        # Check that error message is present (with actual error format)
        error_found = any(
            "[System:" in msg["content"] and "エラー" in msg["content"]
            for msg in final_display
            if msg.get("role") == "assistant" and msg.get("content")
        )
        assert error_found


class TestChatServiceEmptyResponseHandling(
    unittest.IsolatedAsyncioTestCase
):  # Issue #119: async test support
    """Test empty response handling (Issue #79 Review Fix)"""

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_empty_gemini_response_with_no_streaming_content(self, mock_create_provider):
        """Completely empty response should show error message."""
        service = ChatService()

        # Mock provider that returns empty content
        mock_provider = MagicMock()
        mock_provider.name = "gemini"

        async def mock_call_api(*args, **kwargs):
            if False:
                yield {}

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        results = await consume_async_gen(service.process_message("@gemini test"))

        # Should have added error message to display_history
        final_display, final_logic, _chunk = results[-1]
        assert len(final_display) == 2
        assert "[System: Geminiからの応答がありませんでした]" in final_display[1]["content"]

        # Logic history should also have error message
        assert len(final_logic) == 2  # user + gemini
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[1]["content"][0]["type"] == "text"
        assert "応答がありませんでした" in final_logic[1]["content"][0]["content"]

    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_empty_gemini_response_with_partial_streaming_content(self, mock_create_provider):
        """Streaming with partial content should NOT show error message."""
        service = ChatService()

        # Mock provider that returns some text first, then ends
        mock_provider = MagicMock()
        mock_provider.name = "gemini"

        async def mock_call_api(*args, **kwargs):
            yield {"type": "text", "content": "部分的な"}
            yield {"type": "text", "content": "応答"}

        mock_provider.call_api.side_effect = mock_call_api
        mock_create_provider.return_value = mock_provider

        results = await consume_async_gen(service.process_message("@gemini test"))

        # Should NOT have error message (streaming succeeded partially)
        final_display, final_logic, _chunk = results[-1]
        assert len(final_display) == 2
        # Should only have the gemini label + streamed text
        assert "部分的な応答" in final_display[1]["content"]
        assert "[System: Geminiからの応答がありませんでした]" not in final_display[1]["content"]

        # Logic history should have accumulated text (concatenated in one entry)
        assert len(final_logic) == 2
        assert final_logic[1]["content"][0]["content"] == "部分的な応答"


class _FakeAutosaveStore:
    def __init__(self):
        self.calls = []

    def save_autosave(self, user_id, system_prompt, turns):
        self.calls.append(
            {
                "user_id": user_id,
                "system_prompt": system_prompt,
                "turns": turns,
            }
        )


class TestChatServiceAutosave:
    @pytest.mark.asyncio
    async def test_autosave_failure_is_non_fatal(self):
        class _FailingAutosaveStore:
            def save_autosave(self, user_id, system_prompt, turns):
                raise OSError("disk full")

        service = ChatService(
            autosave_store=_FailingAutosaveStore(),
            autosave_user_id="user-1",
            autosave_min_interval_sec=0,
        )

        await consume_async_gen(service.process_message("memo only"))

        assert service.logic_history[-1]["role"] == "user"
        assert service.logic_history[-1]["content"] == [{"type": "text", "content": "memo only"}]

    @pytest.mark.asyncio
    async def test_autosave_failure_emits_warning(self):
        class _FailingAutosaveStore:
            def save_autosave(self, user_id, system_prompt, turns):
                raise OSError("disk full")

        service = ChatService(
            autosave_store=_FailingAutosaveStore(),
            autosave_user_id="user-1",
            autosave_min_interval_sec=0,
        )

        await consume_async_gen(service.process_message("memo only"))

        assert service.consume_autosave_warning() == AUTOSAVE_FAILURE_WARNING
        assert service.consume_autosave_warning() is None

    @pytest.mark.asyncio
    async def test_autosave_updates_on_user_message(self):
        store = _FakeAutosaveStore()
        service = ChatService(
            autosave_store=store,
            autosave_user_id="user-1",
            autosave_min_interval_sec=0,
        )

        await consume_async_gen(service.process_message("memo only"))

        assert len(store.calls) == 1
        assert store.calls[0]["user_id"] == "user-1"
        assert store.calls[0]["turns"][-1]["role"] == "user"
        assert store.calls[0]["turns"][-1]["content"] == [{"type": "text", "content": "memo only"}]

    @patch("multi_llm_chat.chat_service.execute_with_tools_stream")
    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_autosave_updates_on_assistant_finalize(
        self,
        mock_create_provider,
        mock_execute_with_tools_stream,
    ):
        store = _FakeAutosaveStore()
        mock_create_provider.return_value = MagicMock(name="gemini_provider")

        async def fake_stream(*args, **kwargs):
            yield {"type": "text", "content": "done"}
            yield AgenticLoopResult(
                chunks=[{"type": "text", "content": "done"}],
                history_delta=[
                    {
                        "role": "gemini",
                        "content": [{"type": "text", "content": "done"}],
                    }
                ],
                final_text="done",
                iterations_used=1,
                timed_out=False,
                error=None,
            )

        mock_execute_with_tools_stream.side_effect = fake_stream

        service = ChatService(
            autosave_store=store,
            autosave_user_id="user-1",
            autosave_min_interval_sec=0,
        )
        await consume_async_gen(service.process_message("@gemini hi"))

        assert len(store.calls) >= 2
        assert store.calls[-1]["turns"][-1]["role"] == "gemini"
        assert store.calls[-1]["turns"][-1]["content"] == [{"type": "text", "content": "done"}]

    @patch("multi_llm_chat.chat_service.execute_with_tools_stream")
    @patch("multi_llm_chat.chat_service.create_provider")
    @pytest.mark.asyncio
    async def test_autosave_debounce(
        self,
        mock_create_provider,
        mock_execute_with_tools_stream,
    ):
        store = _FakeAutosaveStore()
        now = [0.0]
        sleep_calls = []

        def fake_clock():
            return now[0]

        async def fake_sleep(delay):
            sleep_calls.append(delay)
            now[0] += delay

        mock_create_provider.return_value = MagicMock(name="gemini_provider")

        async def fake_stream(*args, **kwargs):
            yield AgenticLoopResult(
                chunks=[],
                history_delta=[
                    {
                        "role": "gemini",
                        "content": [{"type": "text", "content": "done"}],
                    }
                ],
                final_text="done",
                iterations_used=1,
                timed_out=False,
                error=None,
            )

        mock_execute_with_tools_stream.side_effect = fake_stream

        service = ChatService(
            autosave_store=store,
            autosave_user_id="user-1",
            autosave_min_interval_sec=2.0,
            autosave_clock=fake_clock,
            autosave_sleep=fake_sleep,
        )

        await consume_async_gen(service.process_message("@gemini hi"))
        await asyncio.sleep(0)

        assert len(store.calls) == 2
        assert sleep_calls == [2.0]
        # Debounced flush should persist the latest state (assistant included)
        assert store.calls[-1]["turns"][-1]["role"] == "gemini"

    @pytest.mark.asyncio
    async def test_configure_autosave_cancels_previous_pending_task(self):
        store = _FakeAutosaveStore()
        now = [0.0]
        sleep_calls = []

        def fake_clock():
            return now[0]

        async def fake_sleep(delay):
            sleep_calls.append(delay)
            now[0] += delay

        service = ChatService(
            autosave_store=store,
            autosave_user_id="user-1",
            autosave_min_interval_sec=2.0,
            autosave_clock=fake_clock,
            autosave_sleep=fake_sleep,
        )
        # first save (immediate)
        service.request_autosave()
        # second request schedules delayed save
        service.request_autosave()
        # reconfigure should cancel old pending task
        service.configure_autosave(user_id="user-2", store=store, min_interval_sec=2.0)

        await asyncio.sleep(0)
        assert len(store.calls) == 1

    def test_request_autosave_debounces_with_thread_timer_without_running_loop(self):
        store = _FakeAutosaveStore()
        service = ChatService(
            autosave_store=store,
            autosave_user_id="user-1",
            autosave_min_interval_sec=2.0,
            autosave_clock=lambda: 0.0,
        )

        # first save (immediate)
        service.request_autosave()
        timers = []

        with (
            patch(
                "multi_llm_chat.chat_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running event loop"),
            ),
            patch("multi_llm_chat.chat_service.threading.Timer") as mock_timer,
        ):

            def build_timer(delay, fn):
                timer = MagicMock()
                timer.delay = delay
                timer.fn = fn
                timer.is_alive.return_value = True
                timers.append(timer)
                return timer

            mock_timer.side_effect = build_timer

            # second request should debounce via threading.Timer
            service.request_autosave()

        # should not save immediately on second request
        assert len(store.calls) == 1
        assert len(timers) == 1
        assert timers[0].delay == 2.0
        timers[0].fn()

        # delayed callback should persist once
        assert len(store.calls) == 2
        assert store.calls[-1]["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_request_autosave_cancels_pending_before_immediate_save(self):
        store = _FakeAutosaveStore()
        now = [0.0]
        sleep_calls = []

        def fake_clock():
            return now[0]

        async def fake_sleep(delay):
            sleep_calls.append(delay)
            now[0] += delay

        service = ChatService(
            autosave_store=store,
            autosave_user_id="user-1",
            autosave_min_interval_sec=2.0,
            autosave_clock=fake_clock,
            autosave_sleep=fake_sleep,
        )

        # first save (immediate)
        service.request_autosave()
        # second request schedules delayed save
        service.request_autosave()
        # time passes enough for immediate save branch
        now[0] = 2.1
        service.request_autosave()

        await asyncio.sleep(0)

        # Should be first immediate + third immediate only (no delayed duplicate)
        assert len(store.calls) == 2
        assert sleep_calls == []

    @pytest.mark.asyncio
    async def test_flush_autosave_saves_pending_state_immediately(self):
        store = _FakeAutosaveStore()
        now = [0.0]

        def fake_clock():
            return now[0]

        async def fake_sleep(delay):
            now[0] += delay

        service = ChatService(
            autosave_store=store,
            autosave_user_id="user-1",
            autosave_min_interval_sec=2.0,
            autosave_clock=fake_clock,
            autosave_sleep=fake_sleep,
        )

        service.request_autosave()
        service.request_autosave()
        service.flush_autosave()
        await asyncio.sleep(0)

        assert len(store.calls) == 2

    def test_request_autosave_avoids_reentrant_double_schedule(self):
        store = _FakeAutosaveStore()
        service = ChatService(
            autosave_store=store,
            autosave_user_id="user-1",
            autosave_min_interval_sec=2.0,
            autosave_clock=lambda: 0.0,
        )

        # first save (immediate)
        service.request_autosave()

        class _FakeLoop:
            def __init__(self):
                self.create_task_calls = 0

            def create_task(self, coro):
                self.create_task_calls += 1
                if self.create_task_calls == 1:
                    service.request_autosave()
                coro.close()
                task = MagicMock()
                task.done.return_value = False
                return task

        fake_loop = _FakeLoop()
        with patch("multi_llm_chat.chat_service.asyncio.get_running_loop", return_value=fake_loop):
            service.request_autosave()

        assert fake_loop.create_task_calls == 1


if __name__ == "__main__":
    unittest.main()
