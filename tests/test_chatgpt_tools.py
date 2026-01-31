import unittest
from unittest.mock import MagicMock, patch

import pytest

from multi_llm_chat.llm_provider import (
    ChatGPTProvider,
    mcp_tools_to_openai_format,
    parse_openai_tool_call,
)


async def collect_async_generator(async_gen):
    """Helper to collect async generator results into a list"""
    results = []
    async for item in async_gen:
        results.append(item)
    return results


"""Tests for ChatGPT Tools integration (Issue #80)

TDD Development:
    This test suite follows the Red-Green-Refactor TDD cycle to implement
    tools parameter support for ChatGPTProvider.

    Test Cases:
        Phase 1 (Conversion Functions):
            1. test_mcp_to_openai_tool_conversion
            2. test_mcp_to_openai_with_empty_tools
            3. test_parse_openai_tool_call
            4. test_parse_openai_tool_call_with_invalid_json
        
        Phase 2 (Provider Integration):
            5. test_chatgpt_provider_call_api_with_tools
            6. test_chatgpt_response_with_tool_call
            7. test_chatgpt_streaming_tool_arguments
            8. test_chatgpt_parallel_tool_calls
        
        Phase 3 (Error Handling):
            9. test_invalid_tool_arguments_json
            10. test_missing_tool_call_id
            11. test_tool_call_without_name
"""


class TestOpenAIToolConversion(unittest.TestCase):
    """MCP形式からOpenAI Tools形式への変換をテスト"""

    def setUp(self):
        """テスト用のMCPツール定義を準備"""
        self.mcp_tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            }
        ]

    def test_mcp_to_openai_tool_conversion(self):
        """MCP形式からOpenAI形式への変換が正しく行われること"""
        openai_tools = mcp_tools_to_openai_format(self.mcp_tools)

        self.assertIsNotNone(openai_tools)
        self.assertIsInstance(openai_tools, list)
        self.assertEqual(len(openai_tools), 1)

        tool = openai_tools[0]
        self.assertEqual(tool["type"], "function")
        self.assertIn("function", tool)

        func = tool["function"]
        self.assertEqual(func["name"], "get_weather")
        self.assertEqual(func["description"], self.mcp_tools[0]["description"])
        self.assertEqual(func["parameters"], self.mcp_tools[0]["inputSchema"])

    def test_mcp_to_openai_with_empty_tools(self):
        """空配列やNoneの場合にNoneを返すこと"""
        self.assertIsNone(mcp_tools_to_openai_format(None))
        self.assertIsNone(mcp_tools_to_openai_format([]))

    def test_mcp_to_openai_with_multiple_tools(self):
        """複数のツールが正しく変換されること"""
        multiple_tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "search_web",
                "description": "Search the web",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

        openai_tools = mcp_tools_to_openai_format(multiple_tools)
        self.assertEqual(len(openai_tools), 2)
        self.assertEqual(openai_tools[0]["function"]["name"], "get_weather")
        self.assertEqual(openai_tools[1]["function"]["name"], "search_web")

    def test_mcp_to_openai_skips_tools_without_name(self):
        """名前のないツールはスキップされること"""
        invalid_tools = [
            {"description": "No name tool", "inputSchema": {}},
            {"name": "valid_tool", "description": "Valid", "inputSchema": {}},
        ]

        openai_tools = mcp_tools_to_openai_format(invalid_tools)
        self.assertEqual(len(openai_tools), 1)
        self.assertEqual(openai_tools[0]["function"]["name"], "valid_tool")


class TestOpenAIToolCallParsing(unittest.TestCase):
    """OpenAI tool_callのパース処理をテスト"""

    def test_parse_openai_tool_call(self):
        """完全なtool_call構造を正しくパースできること"""
        tool_call = {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Tokyo"}',
            },
        }

        result = parse_openai_tool_call(tool_call)

        self.assertEqual(result["name"], "get_weather")
        self.assertEqual(result["arguments"], {"location": "Tokyo"})
        self.assertEqual(result["tool_call_id"], "call_abc123")

    def test_parse_openai_tool_call_with_invalid_json(self):
        """不正なJSON argumentsでも例外を投げずに処理すること"""
        tool_call = {
            "id": "call_xyz789",
            "type": "function",
            "function": {
                "name": "broken_tool",
                "arguments": '{"invalid": json syntax',
            },
        }

        result = parse_openai_tool_call(tool_call)

        self.assertEqual(result["name"], "broken_tool")
        self.assertEqual(result["arguments"], {})  # Empty dict on parse failure
        self.assertEqual(result["tool_call_id"], "call_xyz789")

    def test_parse_openai_tool_call_with_empty_arguments(self):
        """空のargumentsを正しく処理すること"""
        tool_call = {
            "id": "call_empty",
            "type": "function",
            "function": {
                "name": "no_args_tool",
                "arguments": "{}",
            },
        }

        result = parse_openai_tool_call(tool_call)
        self.assertEqual(result["arguments"], {})


class TestChatGPTProviderTools(unittest.TestCase):
    """ChatGPTProviderのツール対応をテスト"""

    def setUp(self):
        """テスト用のProviderとツール定義を準備"""
        self.provider = ChatGPTProvider()
        self.mcp_tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]
        self.history = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    @patch("openai.OpenAI")
    @pytest.mark.asyncio
    async def test_chatgpt_provider_call_api_with_tools(self, mock_openai_class):
        """tools引数がOpenAI APIに正しく渡されること"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response with no chunks (just test API call)
        mock_stream = iter([])
        mock_client.chat.completions.create.return_value = mock_stream

        provider = ChatGPTProvider()
        await collect_async_generator(provider.call_api(self.history, tools=self.mcp_tools))

        # Verify API was called with tools
        call_args = mock_client.chat.completions.create.call_args
        self.assertIn("tools", call_args.kwargs)
        self.assertIn("tool_choice", call_args.kwargs)
        self.assertEqual(call_args.kwargs["tool_choice"], "auto")

    @patch("openai.OpenAI")
    @pytest.mark.asyncio
    async def test_chatgpt_response_with_tool_call(self, mock_openai_class):
        """ストリーミング中のtool_callを正しく検出すること"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response with tool_call
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = None

        # Create function mock with proper attributes
        function_mock = MagicMock()
        function_mock.name = "get_weather"
        function_mock.arguments = '{"location": "Tokyo"}'

        tool_call_mock = MagicMock()
        tool_call_mock.index = 0
        tool_call_mock.id = "call_123"
        tool_call_mock.function = function_mock

        chunk1.choices[0].delta.tool_calls = [tool_call_mock]

        mock_stream = iter([chunk1])
        mock_client.chat.completions.create.return_value = mock_stream

        provider = ChatGPTProvider()
        results = await collect_async_generator(
            provider.call_api(self.history, tools=self.mcp_tools)
        )

        # Should emit tool_call event
        tool_calls = [r for r in results if r["type"] == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["content"]["name"], "get_weather")

    @patch("openai.OpenAI")
    @pytest.mark.asyncio
    async def test_chatgpt_streaming_tool_arguments(self, mock_openai_class):
        """複数チャンクに分割されたargumentsを正しく組み立てること"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response with arguments split across chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = None

        function1 = MagicMock()
        function1.name = "get_weather"
        function1.arguments = '{"loc'
        tool_call1 = MagicMock()
        tool_call1.index = 0
        tool_call1.id = "call_456"
        tool_call1.function = function1
        chunk1.choices[0].delta.tool_calls = [tool_call1]

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = None

        function2 = MagicMock()
        function2.name = None
        function2.arguments = 'ation": "T'
        tool_call2 = MagicMock()
        tool_call2.index = 0
        tool_call2.id = None
        tool_call2.function = function2
        chunk2.choices[0].delta.tool_calls = [tool_call2]

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock()
        chunk3.choices[0].delta.content = None

        function3 = MagicMock()
        function3.name = None
        function3.arguments = 'okyo"}'
        tool_call3 = MagicMock()
        tool_call3.index = 0
        tool_call3.id = None
        tool_call3.function = function3
        chunk3.choices[0].delta.tool_calls = [tool_call3]

        mock_stream = iter([chunk1, chunk2, chunk3])
        mock_client.chat.completions.create.return_value = mock_stream

        provider = ChatGPTProvider()
        results = await collect_async_generator(
            provider.call_api(self.history, tools=self.mcp_tools)
        )

        # Should emit single tool_call with complete arguments
        tool_calls = [r for r in results if r["type"] == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["content"]["arguments"]["location"], "Tokyo")

    @patch("openai.OpenAI")
    @pytest.mark.asyncio
    async def test_chatgpt_parallel_tool_calls(self, mock_openai_class):
        """並列ツール呼び出し（複数index）を正しく処理すること"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock parallel tool calls with different indexes
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = None

        function_a = MagicMock()
        function_a.name = "tool_a"
        function_a.arguments = '{"arg": "A"}'
        tool_call_a = MagicMock()
        tool_call_a.index = 0
        tool_call_a.id = "call_A"
        tool_call_a.function = function_a
        chunk1.choices[0].delta.tool_calls = [tool_call_a]

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = None

        function_b = MagicMock()
        function_b.name = "tool_b"
        function_b.arguments = '{"arg": "B"}'
        tool_call_b = MagicMock()
        tool_call_b.index = 1
        tool_call_b.id = "call_B"
        tool_call_b.function = function_b
        chunk2.choices[0].delta.tool_calls = [tool_call_b]

        mock_stream = iter([chunk1, chunk2])
        mock_client.chat.completions.create.return_value = mock_stream

        provider = ChatGPTProvider()
        results = await collect_async_generator(
            provider.call_api(self.history, tools=self.mcp_tools)
        )

        # Should emit two separate tool_calls
        tool_calls = [r for r in results if r["type"] == "tool_call"]
        self.assertEqual(len(tool_calls), 2)

        tool_names = {tc["content"]["name"] for tc in tool_calls}
        self.assertEqual(tool_names, {"tool_a", "tool_b"})


class TestOpenAIToolErrorHandling(unittest.TestCase):
    """OpenAIツール呼び出しのエラーハンドリングをテスト"""

    def test_invalid_tool_arguments_json(self):
        """不正なJSON argumentsでエラーにならないこと"""
        tool_call = {
            "id": "call_err",
            "type": "function",
            "function": {
                "name": "error_tool",
                "arguments": "not a json {{{",
            },
        }

        # Should not raise exception
        result = parse_openai_tool_call(tool_call)
        self.assertEqual(result["name"], "error_tool")
        self.assertEqual(result["arguments"], {})

    def test_missing_tool_call_id(self):
        """tool_call_idが欠けていても処理を続行すること"""
        tool_call = {
            # id missing
            "type": "function",
            "function": {
                "name": "no_id_tool",
                "arguments": '{"key": "value"}',
            },
        }

        result = parse_openai_tool_call(tool_call)
        self.assertEqual(result["name"], "no_id_tool")
        self.assertIsNone(result["tool_call_id"])

    def test_tool_call_without_name(self):
        """nameが欠けている場合の処理"""
        tool_call = {
            "id": "call_noname",
            "type": "function",
            "function": {
                # name missing
                "arguments": '{"key": "value"}',
            },
        }

        result = parse_openai_tool_call(tool_call)
        self.assertIsNone(result["name"])
        self.assertEqual(result["tool_call_id"], "call_noname")


class TestChatGPTFormatHistory(unittest.TestCase):
    """ChatGPTProvider.format_history()のツール対応をテスト"""

    def test_format_history_with_tool_call(self):
        """ツール呼び出しを含む履歴が正しくOpenAI形式に変換されること"""
        history = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "tool_call",
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                        "tool_call_id": "call_123",
                    }
                ],
            },
        ]

        provider = ChatGPTProvider()
        formatted = provider.format_history(history)

        # Should have 2 messages: user + assistant with tool_calls
        self.assertEqual(len(formatted), 2)

        # Check user message
        self.assertEqual(formatted[0]["role"], "user")
        self.assertEqual(formatted[0]["content"], "What's the weather?")

        # Check assistant message with tool_calls
        assistant_msg = formatted[1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertIsNone(assistant_msg["content"])
        self.assertIn("tool_calls", assistant_msg)

        tool_call = assistant_msg["tool_calls"][0]
        self.assertEqual(tool_call["id"], "call_123")
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "get_weather")
        self.assertEqual(tool_call["function"]["arguments"], '{"location": "Tokyo"}')

    def test_format_history_with_tool_result(self):
        """ツール実行結果を含む履歴が正しくOpenAI形式に変換されること"""
        history = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "tool_call",
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                        "tool_call_id": "call_123",
                    }
                ],
            },
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_123",
                        "result": {"temperature": 20, "condition": "sunny"},
                    }
                ],
            },
        ]

        provider = ChatGPTProvider()
        formatted = provider.format_history(history)

        # Should have 3 messages: user + assistant + tool
        self.assertEqual(len(formatted), 3)

        # Check tool message
        tool_msg = formatted[2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_123")
        self.assertEqual(tool_msg["content"], '{"temperature": 20, "condition": "sunny"}')

    def test_format_history_mixed_content(self):
        """テキストとツール呼び出しが混在する履歴を正しく処理すること"""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "chatgpt", "content": "Hi there!"},  # レガシー形式
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "text",
                        "content": "Let me check...",
                    },
                    {
                        "type": "tool_call",
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                        "tool_call_id": "call_456",
                    },
                ],
            },
        ]

        provider = ChatGPTProvider()
        formatted = provider.format_history(history)

        # Should have 4 messages: user + assistant(text) + user + assistant(text+tool_call combined)
        # OpenAI API allows text and tool_calls in the same message
        self.assertEqual(len(formatted), 4)

        # Check legacy text format
        self.assertEqual(formatted[1]["role"], "assistant")
        self.assertEqual(formatted[1]["content"], "Hi there!")

        # Check last message has both text and tool_calls (mixed content is valid)
        last_msg = formatted[3]
        self.assertEqual(last_msg["role"], "assistant")
        self.assertEqual(last_msg["content"], "Let me check...")  # Text is preserved
        self.assertIn("tool_calls", last_msg)
        self.assertEqual(len(last_msg["tool_calls"]), 1)
        self.assertEqual(last_msg["tool_calls"][0]["function"]["name"], "get_weather")

    def test_format_history_skips_incomplete_tool_call(self):
        """tool_call_idやnameが欠けている場合はスキップすること"""
        history = [
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "tool_call",
                        # name missing
                        "arguments": {},
                        "tool_call_id": "call_111",
                    }
                ],
            },
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "tool_call",
                        "name": "valid_tool",
                        "arguments": {},
                        # tool_call_id missing
                    }
                ],
            },
        ]

        provider = ChatGPTProvider()
        formatted = provider.format_history(history)

        # Both should be skipped
        self.assertEqual(len(formatted), 0)

    def test_format_history_skips_tool_result_without_id(self):
        """tool_call_idがないtool_resultはスキップすること"""
        history = [
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "tool_result",
                        # tool_call_id missing
                        "result": {"data": "value"},
                    }
                ],
            }
        ]

        provider = ChatGPTProvider()
        formatted = provider.format_history(history)
        self.assertEqual(len(formatted), 0)

    def test_format_history_parallel_tool_calls(self):
        """同一ターンの複数tool_callsが1メッセージに集約されること"""
        history = [
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "tool_call",
                        "name": "tool_a",
                        "arguments": {"arg": "A"},
                        "tool_call_id": "call_1",
                    },
                    {
                        "type": "tool_call",
                        "name": "tool_b",
                        "arguments": {"arg": "B"},
                        "tool_call_id": "call_2",
                    },
                ],
            }
        ]

        provider = ChatGPTProvider()
        formatted = provider.format_history(history)

        # Should have 1 message with 2 tool_calls
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]["role"], "assistant")
        self.assertIsNone(formatted[0]["content"])
        self.assertEqual(len(formatted[0]["tool_calls"]), 2)

        # Check tool_calls contents
        tool_names = {tc["function"]["name"] for tc in formatted[0]["tool_calls"]}
        self.assertEqual(tool_names, {"tool_a", "tool_b"})
