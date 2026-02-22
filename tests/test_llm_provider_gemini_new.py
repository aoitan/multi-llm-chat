"""Tests for GeminiProvider with New SDK (google.genai)

These tests verify that GeminiProvider works correctly with the new SDK.

Issue: #137 (Phase 2: New SDK Implementation)
Issue: #138 (Phase 3: New SDK as Default)
Issue: #139 (Phase 4: Legacy SDK removed - these tests now always run)
"""

import asyncio
import unittest
from unittest.mock import Mock, patch

import pytest

from multi_llm_chat.providers.gemini import GeminiProvider


class TestGeminiProviderNewSDK:
    """Integration tests for GeminiProvider using new SDK"""

    def setup_method(self):
        """Setup common test data"""
        self.history = [{"role": "user", "content": "Hello"}]

    @patch("google.genai.Client")
    @pytest.mark.asyncio
    async def test_call_api_yields_text_chunks_new_sdk(self, mock_client_class):
        """call_api should yield text chunks with new SDK"""
        # Mock client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock response chunks
        mock_chunk1 = Mock()
        mock_chunk1.text = "Hello"
        mock_chunk1.parts = []
        mock_chunk2 = Mock()
        mock_chunk2.text = " world"
        mock_chunk2.parts = []

        mock_client.models.generate_content_stream.return_value = iter([mock_chunk1, mock_chunk2])

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")
        results = []
        async for event in provider.call_api(self.history):
            results.append(event)

        assert len(results) == 2
        assert results[0] == {"type": "text", "content": "Hello"}
        assert results[1] == {"type": "text", "content": " world"}
        mock_client.models.generate_content_stream.assert_called_once()

    @patch("google.genai.Client")
    @pytest.mark.asyncio
    async def test_call_api_with_system_instruction_new_sdk(self, mock_client_class):
        """call_api should pass system instruction to new SDK"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_chunk = Mock()
        mock_chunk.text = "Response"
        mock_chunk.parts = []
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")
        results = []
        async for event in provider.call_api(self.history, system_prompt="You are helpful"):
            results.append(event)

        # Verify system instruction was passed
        call_args = mock_client.models.generate_content_stream.call_args
        assert call_args is not None
        config = call_args.kwargs.get("config")
        assert config is not None
        assert config.get("system_instruction") == "You are helpful"

    @patch("google.genai.Client")
    @pytest.mark.asyncio
    async def test_call_api_with_tools_new_sdk(self, mock_client_class):
        """call_api should pass tools to new SDK"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_chunk = Mock()
        mock_chunk.text = "Response"
        mock_chunk.parts = []
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")
        # MCP tool format (not OpenAI format)
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

        results = []
        async for event in provider.call_api(self.history, tools=tools):
            results.append(event)

        # Verify tools were passed
        call_args = mock_client.models.generate_content_stream.call_args
        assert call_args is not None
        config = call_args.kwargs.get("config")
        assert config is not None
        assert "tools" in config

    @patch("google.genai.Client")
    @pytest.mark.asyncio
    async def test_tool_serialization_new_sdk(self, mock_client_class):
        """Tools should be properly serialized for new SDK (Critical: Issue #141)"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_chunk = Mock()
        mock_chunk.text = "Calling tool"
        mock_chunk.parts = []
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        async for _ in provider.call_api(self.history, tools=tools):
            pass

        # Verify tools were converted to new SDK format
        call_args = mock_client.models.generate_content_stream.call_args
        config = call_args.kwargs.get("config")
        assert "tools" in config

        # Verify the tools are google.genai.types.Tool instances (not legacy SDK)
        tools_arg = config["tools"]
        assert len(tools_arg) == 1
        # New SDK accepts Tool objects from google.genai.types
        # This test will fail if we're passing legacy SDK Tool objects
        from google.genai.types import Tool as NewTool

        assert isinstance(tools_arg[0], NewTool), (
            f"Expected google.genai.types.Tool, got {type(tools_arg[0]).__module__}."
            f"{type(tools_arg[0]).__name__}"
        )

    @patch("google.genai.Client")
    @pytest.mark.asyncio
    async def test_model_caching_new_sdk(self, mock_client_class):
        """Model proxy should be cached across calls"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_chunk = Mock()
        mock_chunk.text = "Response"
        mock_chunk.parts = []
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")

        # First call
        async for _ in provider.call_api(self.history):
            pass

        # Second call should use cached model
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])
        async for _ in provider.call_api(self.history):
            pass

        # Client should only be created once
        assert mock_client_class.call_count == 1


class TestMCPToolsConversion(unittest.TestCase):
    """Tests for MCP tool conversion and schema sanitization (Issue #139 migration)"""

    def test_mcp_tools_to_gemini_format_basic(self):
        """mcp_tools_to_gemini_format should convert MCP tools to Gemini Tool format"""
        from google.genai.types import Tool

        from multi_llm_chat.providers.gemini import mcp_tools_to_gemini_format

        mcp_tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        }
                    },
                    "required": ["location"],
                },
            }
        ]

        gemini_tools = mcp_tools_to_gemini_format(mcp_tools)
        self.assertIsInstance(gemini_tools, list)
        self.assertEqual(len(gemini_tools), 1)
        self.assertIsInstance(gemini_tools[0], Tool)

        decls = gemini_tools[0].function_declarations
        self.assertIsNotNone(decls)
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0].name, "get_weather")
        self.assertEqual(decls[0].description, mcp_tools[0]["description"])

    def test_mcp_tools_returns_none_for_empty(self):
        """mcp_tools_to_gemini_format should return None for empty list"""
        from multi_llm_chat.providers.gemini import mcp_tools_to_gemini_format

        self.assertIsNone(mcp_tools_to_gemini_format([]))
        self.assertIsNone(mcp_tools_to_gemini_format(None))

    def test_mcp_tools_schema_dollar_field_removed(self):
        """$schema field should be removed before passing to Gemini API"""
        from multi_llm_chat.providers.gemini import _sanitize_schema_for_gemini

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"location": {"type": "string"}},
        }
        result = _sanitize_schema_for_gemini(schema)
        self.assertNotIn("$schema", result)
        self.assertEqual(result["type"], "object")

    def test_mcp_tools_validation_fields_removed(self):
        """Validation fields (minItems, maxItems, pattern, etc.) should be removed"""
        from multi_llm_chat.providers.gemini import _sanitize_schema_for_gemini

        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                },
                "name": {
                    "type": "string",
                    "pattern": "^[a-z]+$",
                    "format": "email",
                },
            },
            "required": ["tags"],
        }
        result = _sanitize_schema_for_gemini(schema)
        # type, properties, required should be preserved
        self.assertEqual(result["type"], "object")
        self.assertIn("properties", result)
        self.assertIn("required", result)
        # Validation fields should be removed
        tags_schema = result["properties"]["tags"]
        self.assertNotIn("minItems", tags_schema)
        self.assertNotIn("maxItems", tags_schema)
        name_schema = result["properties"]["name"]
        self.assertNotIn("pattern", name_schema)
        self.assertNotIn("format", name_schema)


class TestGeminiProviderToolCallStreaming(unittest.TestCase):
    """Tests for GeminiProvider tool call streaming with new SDK (Issue #139 migration)"""

    def setUp(self):
        self.history = [{"role": "user", "content": "What's the weather?"}]

    def _make_tool_chunk(self, name=None, args=None, index=None):
        """Helper: create a mock chunk with function_call part"""
        from unittest.mock import MagicMock

        fc = MagicMock()
        fc.name = name
        fc.args = args
        part = MagicMock(spec=["function_call", "index"])
        part.function_call = fc
        part.index = index
        chunk = MagicMock()
        chunk.parts = [part]
        return chunk

    @patch("google.genai.Client")
    def test_call_api_yields_single_tool_call(self, mock_client_class):
        """call_api should yield tool_call event for a single tool call (name then args)"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        chunk1 = self._make_tool_chunk(name="get_weather", args=None, index=None)
        chunk2 = self._make_tool_chunk(name=None, args={"location": "Tokyo"}, index=None)
        mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")
        tools = [{"name": "get_weather", "description": "test", "inputSchema": {}}]

        results = asyncio.run(self._collect(provider.call_api(self.history, tools=tools)))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "tool_call")
        self.assertEqual(results[0]["content"]["name"], "get_weather")
        self.assertEqual(results[0]["content"]["arguments"], {"location": "Tokyo"})

    @patch("google.genai.Client")
    def test_call_api_no_tool_calls_emitted_on_error(self, mock_client_class):
        """Pending tool calls should NOT be emitted if stream raises ValueError"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        chunk_name = self._make_tool_chunk(name="get_weather", args=None, index=None)

        def failing_stream():
            yield chunk_name
            raise ValueError("API error")

        mock_client.models.generate_content_stream.return_value = failing_stream()

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")
        tools = [{"name": "get_weather", "description": "test", "inputSchema": {}}]

        with self.assertRaises(ValueError):
            asyncio.run(self._collect(provider.call_api(self.history, tools=tools)))

    @patch("google.genai.Client")
    def test_call_api_handles_mixed_text_and_tool_call(self, mock_client_class):
        """call_api should yield both text and tool_call events from mixed stream"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Text chunk
        text_chunk = Mock()
        text_chunk.parts = []
        text_chunk.text = "I'll check the weather."

        # Tool call chunks
        tc_chunk1 = self._make_tool_chunk(name="get_weather", args=None, index=None)
        tc_chunk2 = self._make_tool_chunk(name=None, args={"location": "Tokyo"}, index=None)

        mock_client.models.generate_content_stream.return_value = iter(
            [text_chunk, tc_chunk1, tc_chunk2]
        )

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")
        tools = [{"name": "get_weather", "description": "test", "inputSchema": {}}]

        results = asyncio.run(self._collect(provider.call_api(self.history, tools=tools)))

        types = [r["type"] for r in results]
        self.assertIn("text", types)
        self.assertIn("tool_call", types)
        tool_results = [r for r in results if r["type"] == "tool_call"]
        self.assertEqual(tool_results[0]["content"]["name"], "get_weather")

    @patch("google.genai.Client")
    def test_call_api_tool_call_order_preserved(self, mock_client_class):
        """Multiple tool calls should be yielded in order"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Two sequential tool calls
        tc1_name = self._make_tool_chunk(name="search", args=None, index=None)
        tc1_args = self._make_tool_chunk(name=None, args={"query": "weather"}, index=None)
        tc2_name = self._make_tool_chunk(name="calc", args=None, index=None)
        tc2_args = self._make_tool_chunk(name=None, args={"expr": "1+1"}, index=None)

        mock_client.models.generate_content_stream.return_value = iter(
            [tc1_name, tc1_args, tc2_name, tc2_args]
        )

        provider = GeminiProvider(api_key="test-key", model_name="gemini-2.0-flash-exp")
        tools = [
            {"name": "search", "description": "test", "inputSchema": {}},
            {"name": "calc", "description": "test", "inputSchema": {}},
        ]

        results = asyncio.run(self._collect(provider.call_api(self.history, tools=tools)))

        tool_calls = [r for r in results if r["type"] == "tool_call"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["content"]["name"], "search")
        self.assertEqual(tool_calls[1]["content"]["name"], "calc")

    @staticmethod
    async def _collect(gen):
        results = []
        async for item in gen:
            results.append(item)
        return results


class TestGeminiFormatHistory(unittest.TestCase):
    """Tests for GeminiProvider.format_history (SDK-independent, Issue #139 migration)"""

    def test_format_history_with_full_tool_cycle(self):
        """format_history should handle user → gemini (tool_call) → tool → model cycle"""
        import json

        from multi_llm_chat.providers.gemini import GeminiProvider

        logic_history = [
            {"role": "user", "content": [{"type": "text", "content": "What's the weather?"}]},
            {
                "role": "gemini",
                "content": [
                    {
                        "type": "tool_call",
                        "content": {
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo"},
                            "tool_call_id": "tc_001",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "tc_001",
                        "content": '{"temperature": "25°C"}',
                        "name": "get_weather",
                    }
                ],
            },
        ]

        formatted = GeminiProvider.format_history(logic_history)

        self.assertEqual(len(formatted), 3)
        self.assertEqual(formatted[0]["role"], "user")
        self.assertEqual(formatted[1]["role"], "model")
        fc_part = formatted[1]["parts"][0]["function_call"]
        self.assertEqual(fc_part["name"], "get_weather")
        self.assertEqual(formatted[2]["role"], "function")
        fr_part = formatted[2]["parts"][0]["function_response"]
        self.assertEqual(fr_part["name"], "get_weather")
        self.assertEqual(fr_part["response"], json.loads('{"temperature": "25°C"}'))

    def test_format_history_wraps_non_json_tool_result(self):
        """format_history should wrap non-JSON string tool results in {result: ...}"""
        from multi_llm_chat.providers.gemini import GeminiProvider

        logic_history = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "tc_1",
                        "content": "plain text response",
                        "name": "search",
                    }
                ],
            }
        ]
        formatted = GeminiProvider.format_history(logic_history)
        fr = formatted[0]["parts"][0]["function_response"]
        self.assertEqual(fr["response"], {"result": "plain text response"})

    def test_format_history_wraps_non_object_json_tool_result(self):
        """format_history should wrap non-object JSON tool results in {result: ...}"""
        from multi_llm_chat.providers.gemini import GeminiProvider

        logic_history = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "tc_2",
                        "content": "[1, 2, 3]",
                        "name": "list_items",
                    }
                ],
            }
        ]
        formatted = GeminiProvider.format_history(logic_history)
        fr = formatted[0]["parts"][0]["function_response"]
        self.assertEqual(fr["response"], {"result": [1, 2, 3]})

    def test_format_history_handles_unexpected_content_types(self):
        """format_history should convert non-list/non-string content to string"""
        from multi_llm_chat.providers.gemini import GeminiProvider

        logic_history = [
            {"role": "user", "content": 123},
            {"role": "gemini", "content": True},
        ]
        formatted = GeminiProvider.format_history(logic_history)
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["parts"][0]["text"], "123")
        self.assertEqual(formatted[1]["parts"][0]["text"], "True")

    def test_format_history_handles_none_content(self):
        """format_history should handle None content gracefully"""
        from multi_llm_chat.providers.gemini import GeminiProvider

        logic_history = [
            {"role": "user", "content": None},
            {"role": "gemini", "content": None},
        ]
        formatted = GeminiProvider.format_history(logic_history)
        self.assertEqual(formatted, [])


class TestParseToolResponsePayload(unittest.TestCase):
    """Tests for _parse_tool_response_payload (SDK-independent, Issue #139 migration)"""

    def test_handles_none_returns_empty_dict(self):
        """None payload should return empty dict"""
        from multi_llm_chat.providers.gemini import _parse_tool_response_payload

        self.assertEqual(_parse_tool_response_payload(None), {})

    def test_handles_dict_passthrough(self):
        """Dict payload should be returned as-is"""
        from multi_llm_chat.providers.gemini import _parse_tool_response_payload

        payload = {"key": "value"}
        self.assertEqual(_parse_tool_response_payload(payload), payload)

    def test_handles_invalid_json_string(self):
        """Non-JSON string should be wrapped in {result: ...}"""
        from multi_llm_chat.providers.gemini import _parse_tool_response_payload

        result = _parse_tool_response_payload("not valid json")
        self.assertEqual(result, {"result": "not valid json"})

    def test_handles_whitelisted_scalar_types(self):
        """int, float, list, bool should be wrapped in {result: ...}"""
        from multi_llm_chat.providers.gemini import _parse_tool_response_payload

        self.assertEqual(_parse_tool_response_payload(42), {"result": 42})
        self.assertEqual(_parse_tool_response_payload(3.14), {"result": 3.14})
        self.assertEqual(_parse_tool_response_payload([1, 2, 3]), {"result": [1, 2, 3]})
        self.assertEqual(_parse_tool_response_payload(True), {"result": True})

    def test_raises_for_unserializable_type(self):
        """Object that raises TypeError in str() should raise TypeError"""
        from multi_llm_chat.providers.gemini import _parse_tool_response_payload

        class UnserializableObject:
            def __str__(self):
                raise TypeError("Cannot serialize")

        with self.assertRaises(TypeError, msg="cannot be safely converted to dict"):
            _parse_tool_response_payload(UnserializableObject())


class TestGeminiToolCallAssembler(unittest.TestCase):
    """Tests for GeminiToolCallAssembler (SDK-independent, Issue #139 migration)"""

    def test_tool_call_with_empty_args_emits_immediately(self):
        """Tool call with empty {} args should emit when args arrive"""
        from unittest.mock import MagicMock

        from multi_llm_chat.providers.gemini import GeminiToolCallAssembler

        assembler = GeminiToolCallAssembler()

        # Name-only chunk (sequential, index=None)
        part_name = MagicMock()
        part_name.index = None
        fc_name = MagicMock()
        fc_name.name = "get_current_time"
        fc_name.args = None
        result1 = assembler.process_function_call(part_name, fc_name)
        self.assertIsNone(result1, "Should not emit with name only")

        # Empty args chunk
        part_args = MagicMock()
        part_args.index = None
        fc_args = MagicMock()
        fc_args.name = None
        fc_args.args = {}
        result2 = assembler.process_function_call(part_args, fc_args)

        self.assertIsNotNone(result2, "Should emit immediately when empty args arrive")
        self.assertEqual(result2["type"], "tool_call")
        self.assertEqual(result2["content"]["name"], "get_current_time")
        self.assertEqual(result2["content"]["arguments"], {})

    def test_tool_call_with_none_args_finalized_at_stream_end(self):
        """Tool call with args=None (not yet received) should be emitted at finalize"""
        from unittest.mock import MagicMock

        from multi_llm_chat.providers.gemini import GeminiToolCallAssembler

        assembler = GeminiToolCallAssembler()

        part = MagicMock()
        part.index = None
        fc = MagicMock()
        fc.name = "list_users"
        fc.args = None
        result = assembler.process_function_call(part, fc)
        self.assertIsNone(result, "Should not emit immediately when args=None")

        pending = list(assembler.finalize_pending_calls())
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["content"]["name"], "list_users")


if __name__ == "__main__":
    unittest.main()
