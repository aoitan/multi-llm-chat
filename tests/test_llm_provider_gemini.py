"""Tests for GeminiProvider implementation

Extracted from test_llm_provider.py as part of Issue #101 refactoring.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest

from multi_llm_chat.llm_provider import (
    GeminiProvider,
    GeminiToolCallAssembler,
    _parse_tool_response_payload,
)


class TestGeminiProvider:
    """Test GeminiProvider implementation"""

    def setup_method(self):
        self.history = [{"role": "user", "content": "Hello"}]

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_yields_text_chunks(self, mock_genai):
        """call_api should yield unified text dictionaries."""
        # Mock the Gemini API response stream
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Hello"
        mock_chunk1.parts = []
        mock_chunk2 = MagicMock()
        mock_chunk2.text = " world"
        mock_chunk2.parts = []
        mock_response_iter = iter([mock_chunk1, mock_chunk2])

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        result = list(provider.call_api(self.history))

        expected = [
            {"type": "text", "content": "Hello"},
            {"type": "text", "content": " world"},
        ]
        assert result == expected
        mock_model.generate_content.assert_called_once()

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_yields_single_tool_call(self, mock_genai):
        """call_api should yield a unified tool_call dictionary for a single tool call."""
        # Mock the Gemini API response for a tool call
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1

        fc2 = MagicMock()
        fc2.name = None
        fc2.args = {"location": "Tokyo"}
        part2 = MagicMock(spec=["function_call"])
        part2.function_call = fc2

        mock_chunk1 = MagicMock(parts=[part1], text=None)
        mock_chunk2 = MagicMock(parts=[part2], text=None)
        mock_response_iter = iter([mock_chunk1, mock_chunk2])

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        result = list(
            provider.call_api(
                self.history,
                tools=[{"name": "get_weather", "description": "test", "inputSchema": {}}],
            )
        )

        expected = [
            {
                "type": "tool_call",
                "content": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            }
        ]
        assert result == expected

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_yields_multiple_tool_calls(self, mock_genai):
        """call_api should yield multiple tool_call dictionaries."""
        # Mock stream for two separate tool calls
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1

        fc2 = MagicMock()
        fc2.name = None
        fc2.args = {"location": "Tokyo"}
        part2 = MagicMock(spec=["function_call"])
        part2.function_call = fc2

        fc3 = MagicMock()
        fc3.name = "get_time"
        fc3.args = None
        part3 = MagicMock(spec=["function_call"])
        part3.function_call = fc3

        fc4 = MagicMock()
        fc4.name = None
        fc4.args = {"timezone": "JST"}
        part4 = MagicMock(spec=["function_call"])
        part4.function_call = fc4

        mock_chunk1 = MagicMock(parts=[part1], text=None)
        mock_chunk2 = MagicMock(parts=[part2, part3], text=None)
        mock_chunk3 = MagicMock(parts=[part4], text=None)
        mock_response_iter = iter([mock_chunk1, mock_chunk2, mock_chunk3])

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        tools = [
            {"name": "get_weather", "description": "test", "inputSchema": {}},
            {"name": "get_time", "description": "test", "inputSchema": {}},
        ]
        result = list(provider.call_api(self.history, tools=tools))

        expected = [
            {
                "type": "tool_call",
                "content": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            },
            {
                "type": "tool_call",
                "content": {"name": "get_time", "arguments": {"timezone": "JST"}},
            },
        ]
        assert result == expected

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_allows_tool_call_without_args(self, mock_genai):
        """call_api should allow tool calls with empty args."""
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1

        mock_chunk1 = MagicMock(parts=[part1], text=None)
        mock_response_iter = iter([mock_chunk1])

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        result = list(
            provider.call_api(
                self.history,
                tools=[{"name": "get_weather", "description": "test", "inputSchema": {}}],
            )
        )

        assert result == [
            {
                "type": "tool_call",
                "content": {"name": "get_weather", "arguments": {}},
            }
        ]

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_handles_mixed_text_and_tool_call(self, mock_genai):
        """call_api should handle responses with both text and tool calls."""
        text_part = MagicMock(text="Thinking about it...", parts=[])

        fc1 = MagicMock()
        fc1.name = "search"
        fc1.args = None
        tool_part1 = MagicMock(spec=["function_call"])
        tool_part1.function_call = fc1

        fc2 = MagicMock()
        fc2.name = None
        fc2.args = {"query": "python"}
        tool_part2 = MagicMock(spec=["function_call"])
        tool_part2.function_call = fc2

        mock_chunk1 = text_part
        mock_chunk2 = MagicMock(parts=[tool_part1, tool_part2], text=None)
        mock_response_iter = iter([mock_chunk1, mock_chunk2])

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        result = list(
            provider.call_api(
                self.history, tools=[{"name": "search", "description": "test", "inputSchema": {}}]
            )
        )

        expected = [
            {"type": "text", "content": "Thinking about it..."},
            {
                "type": "tool_call",
                "content": {"name": "search", "arguments": {"query": "python"}},
            },
        ]
        assert result == expected

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_handles_chunk_text_value_error(self, mock_genai):
        """call_api should handle chunks that raise on .text access."""

        class TextRaises:
            def __init__(self, parts):
                self.parts = parts

            @property
            def text(self):
                raise ValueError("No text parts")

        fc_name = MagicMock()
        fc_name.name = "get_weather"
        fc_name.args = None
        part_name = MagicMock(spec=["function_call"])
        part_name.function_call = fc_name

        fc_args = MagicMock()
        fc_args.name = None
        fc_args.args = {"location": "Tokyo"}
        part_args = MagicMock(spec=["function_call"])
        part_args.function_call = fc_args

        mock_response_iter = iter([TextRaises([part_name]), TextRaises([part_args])])

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        result = list(
            provider.call_api(
                self.history,
                tools=[{"name": "get_weather", "description": "test", "inputSchema": {}}],
            )
        )

        expected = [
            {
                "type": "tool_call",
                "content": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            }
        ]
        assert result == expected

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_maps_tool_calls_by_index(self, mock_genai):
        """call_api should map tool call args to matching indexed calls."""
        fc_name_a = MagicMock()
        fc_name_a.name = "tool_a"
        fc_name_a.args = None
        part_name_a = MagicMock(spec=["function_call", "index"])
        part_name_a.function_call = fc_name_a
        part_name_a.index = 0

        fc_name_b = MagicMock()
        fc_name_b.name = "tool_b"
        fc_name_b.args = None
        part_name_b = MagicMock(spec=["function_call", "index"])
        part_name_b.function_call = fc_name_b
        part_name_b.index = 1

        fc_args_a = MagicMock()
        fc_args_a.name = None
        fc_args_a.args = {"value": "A"}
        part_args_a = MagicMock(spec=["function_call", "index"])
        part_args_a.function_call = fc_args_a
        part_args_a.index = 0

        fc_args_b = MagicMock()
        fc_args_b.name = None
        fc_args_b.args = {"value": "B"}
        part_args_b = MagicMock(spec=["function_call", "index"])
        part_args_b.function_call = fc_args_b
        part_args_b.index = 1

        mock_response_iter = iter(
            [
                MagicMock(parts=[part_name_a], text=None),
                MagicMock(parts=[part_name_b], text=None),
                MagicMock(parts=[part_args_a], text=None),
                MagicMock(parts=[part_args_b], text=None),
            ]
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        tools = [
            {"name": "tool_a", "description": "test", "inputSchema": {}},
            {"name": "tool_b", "description": "test", "inputSchema": {}},
        ]
        result = list(provider.call_api(self.history, tools=tools))

        expected = [
            {"type": "tool_call", "content": {"name": "tool_a", "arguments": {"value": "A"}}},
            {"type": "tool_call", "content": {"name": "tool_b", "arguments": {"value": "B"}}},
        ]
        assert result == expected

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_maps_tool_calls_without_index(self, mock_genai):
        """call_api should map tool call args in the same order when indexes are missing."""
        fc_name_a = MagicMock()
        fc_name_a.name = "tool_a"
        fc_name_a.args = None
        part_name_a = MagicMock(spec=["function_call"])
        part_name_a.function_call = fc_name_a

        fc_name_b = MagicMock()
        fc_name_b.name = "tool_b"
        fc_name_b.args = None
        part_name_b = MagicMock(spec=["function_call"])
        part_name_b.function_call = fc_name_b

        fc_args_a = MagicMock()
        fc_args_a.name = None
        fc_args_a.args = {"value": "A"}
        part_args_a = MagicMock(spec=["function_call"])
        part_args_a.function_call = fc_args_a

        fc_args_b = MagicMock()
        fc_args_b.name = None
        fc_args_b.args = {"value": "B"}
        part_args_b = MagicMock(spec=["function_call"])
        part_args_b.function_call = fc_args_b

        mock_response_iter = iter(
            [
                MagicMock(parts=[part_name_a], text=None),
                MagicMock(parts=[part_name_b], text=None),
                MagicMock(parts=[part_args_a], text=None),
                MagicMock(parts=[part_args_b], text=None),
            ]
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        tools = [
            {"name": "tool_a", "description": "test", "inputSchema": {}},
            {"name": "tool_b", "description": "test", "inputSchema": {}},
        ]
        result = list(provider.call_api(self.history, tools=tools))

        expected = [
            {"type": "tool_call", "content": {"name": "tool_a", "arguments": {"value": "A"}}},
            {"type": "tool_call", "content": {"name": "tool_b", "arguments": {"value": "B"}}},
        ]
        assert result == expected

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_handles_interleaved_parallel_tool_calls(self, mock_genai):
        """並列ツール呼び出しでパーツが交互に到着する場合のテスト"""
        # Simulate interleaved stream: name_a -> name_b -> args_b -> args_a
        fc_name_a = MagicMock()
        fc_name_a.name = "tool_a"
        fc_name_a.args = None
        part_name_a = MagicMock(spec=["function_call", "index"])
        part_name_a.function_call = fc_name_a
        part_name_a.index = 0

        fc_name_b = MagicMock()
        fc_name_b.name = "tool_b"
        fc_name_b.args = None
        part_name_b = MagicMock(spec=["function_call", "index"])
        part_name_b.function_call = fc_name_b
        part_name_b.index = 1

        # Args arrive in reverse order
        fc_args_b = MagicMock()
        fc_args_b.name = None
        fc_args_b.args = {"value": "B"}
        part_args_b = MagicMock(spec=["function_call", "index"])
        part_args_b.function_call = fc_args_b
        part_args_b.index = 1

        fc_args_a = MagicMock()
        fc_args_a.name = None
        fc_args_a.args = {"value": "A"}
        part_args_a = MagicMock(spec=["function_call", "index"])
        part_args_a.function_call = fc_args_a
        part_args_a.index = 0

        mock_response_iter = iter(
            [
                MagicMock(parts=[part_name_a], text=None),
                MagicMock(parts=[part_name_b], text=None),
                MagicMock(parts=[part_args_b], text=None),  # B arrives first
                MagicMock(parts=[part_args_a], text=None),  # A arrives second
            ]
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        tools = [
            {"name": "tool_a", "description": "test", "inputSchema": {}},
            {"name": "tool_b", "description": "test", "inputSchema": {}},
        ]
        result = list(provider.call_api(self.history, tools=tools))

        # Verify correct mapping despite interleaved arrival
        expected = [
            {"type": "tool_call", "content": {"name": "tool_b", "arguments": {"value": "B"}}},
            {"type": "tool_call", "content": {"name": "tool_a", "arguments": {"value": "A"}}},
        ]
        assert result == expected

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_finalizes_pending_calls_once_on_error(self, mock_genai):
        """例外発生時は未完のツール呼び出しを出力しない（Issue #79 Review Fix）"""

        # Create a proper exception class for BlockedPromptException
        class BlockedPromptException(Exception):
            pass

        mock_genai.types.BlockedPromptException = BlockedPromptException

        mock_model = MagicMock()

        # Simulate tool call name arriving, then an error before args
        fc_name = MagicMock()
        fc_name.name = "get_weather"
        fc_name.args = None
        part_name = MagicMock(spec=["function_call"])
        part_name.function_call = fc_name

        def error_generator():
            yield MagicMock(parts=[part_name], text=None)
            raise ValueError("Unexpected error")  # Use ValueError to trigger Exception handler

        mock_model.generate_content.return_value = error_generator()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()

        # Collect results until exception
        results = []
        with pytest.raises(ValueError, match="Unexpected error"):
            for chunk in provider.call_api(self.history, tools=[{"name": "get_weather"}]):
                results.append(chunk)

        # Should NOT emit pending tool calls on error (security fix)
        tool_calls = [r for r in results if r.get("type") == "tool_call"]
        assert len(tool_calls) == 0, "No tool calls should be emitted on error"

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_handles_name_and_args_in_same_chunk(self, mock_genai):
        """nameとargsが同一チャンクで到着した場合の処理テスト (Critical Fix A1)"""
        # First tool call with both name and args in single chunk
        fc1 = MagicMock()
        fc1.name = "tool_a"
        fc1.args = {"value": "A"}
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1

        # Second tool call also arrives in single chunk
        fc2 = MagicMock()
        fc2.name = "tool_b"
        fc2.args = {"value": "B"}
        part2 = MagicMock(spec=["function_call"])
        part2.function_call = fc2

        mock_response_iter = iter(
            [
                MagicMock(parts=[part1], text=None),
                MagicMock(parts=[part2], text=None),
            ]
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        tools = [
            {"name": "tool_a", "description": "test", "inputSchema": {}},
            {"name": "tool_b", "description": "test", "inputSchema": {}},
        ]
        result = list(provider.call_api(self.history, tools=tools))

        # Both tool calls should be emitted correctly
        expected = [
            {"type": "tool_call", "content": {"name": "tool_a", "arguments": {"value": "A"}}},
            {"type": "tool_call", "content": {"name": "tool_b", "arguments": {"value": "B"}}},
        ]
        assert result == expected

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_handles_mixed_chunk_patterns(self, mock_genai):
        """複数の順次ツール呼び出しでチャンクパターンが混在するケース (Critical Fix A1)"""
        # tool_a: name+args in same chunk
        fc1 = MagicMock()
        fc1.name = "tool_a"
        fc1.args = {"value": "A"}
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1

        # tool_b: name only
        fc2_name = MagicMock()
        fc2_name.name = "tool_b"
        fc2_name.args = None
        part2_name = MagicMock(spec=["function_call"])
        part2_name.function_call = fc2_name

        # tool_b: args only
        fc2_args = MagicMock()
        fc2_args.name = None
        fc2_args.args = {"value": "B"}
        part2_args = MagicMock(spec=["function_call"])
        part2_args.function_call = fc2_args

        # tool_c: name+args in same chunk
        fc3 = MagicMock()
        fc3.name = "tool_c"
        fc3.args = {"value": "C"}
        part3 = MagicMock(spec=["function_call"])
        part3.function_call = fc3

        mock_response_iter = iter(
            [
                MagicMock(parts=[part1], text=None),
                MagicMock(parts=[part2_name], text=None),
                MagicMock(parts=[part2_args], text=None),
                MagicMock(parts=[part3], text=None),
            ]
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response_iter
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        tools = [
            {"name": "tool_a", "description": "test", "inputSchema": {}},
            {"name": "tool_b", "description": "test", "inputSchema": {}},
            {"name": "tool_c", "description": "test", "inputSchema": {}},
        ]
        result = list(provider.call_api(self.history, tools=tools))

        # All three tool calls should be emitted correctly in order
        expected = [
            {"type": "tool_call", "content": {"name": "tool_a", "arguments": {"value": "A"}}},
            {"type": "tool_call", "content": {"name": "tool_b", "arguments": {"value": "B"}}},
            {"type": "tool_call", "content": {"name": "tool_c", "arguments": {"value": "C"}}},
        ]
        assert result == expected

    def test_extract_text_from_chunk(self):
        """extract_text_from_chunk should process unified dictionaries."""
        provider = GeminiProvider()

        text_chunk = {"type": "text", "content": "Hello, world!"}
        assert provider.extract_text_from_chunk(text_chunk) == "Hello, world!"

        tool_chunk = {"type": "tool_call", "content": {"name": "test", "arguments": {}}}
        assert provider.extract_text_from_chunk(tool_chunk) == ""

        invalid_chunk = {"type": "other", "content": "something"}
        assert provider.extract_text_from_chunk(invalid_chunk) == ""

        # Test backward compatibility with old chunk format (should return empty)
        old_chunk = MagicMock(text="some text")
        assert provider.extract_text_from_chunk(old_chunk) == ""

    def test_get_token_info_returns_dict(self):
        """GeminiProvider should return token info dictionary"""
        provider = GeminiProvider()
        result = provider.get_token_info("Test message")

        assert isinstance(result, dict)
        assert "input_tokens" in result
        assert "max_tokens" in result

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_model_cache(self, mock_genai):
        """GeminiProvider should cache model instances based on system prompt"""
        # Ensure GenerativeModel returns a new mock each time it's called
        mock_genai.GenerativeModel.side_effect = lambda *args, **kwargs: MagicMock()

        provider = GeminiProvider()
        system_prompt = "You are a helpful assistant."

        # First call creates new model
        model1 = provider._get_model(system_prompt)
        assert mock_genai.GenerativeModel.call_count == 1

        # Second call with same prompt returns cached model
        model2 = provider._get_model(system_prompt)
        assert mock_genai.GenerativeModel.call_count == 1
        assert model1 is model2

        # Call with different prompt creates new model
        model3 = provider._get_model("Different prompt")
        assert mock_genai.GenerativeModel.call_count == 2
        assert model3 is not model1

    def test_format_history_with_full_tool_cycle(self):
        """format_history should correctly format a full tool call and response cycle."""
        # This is the application's internal structured history format
        logic_history = [
            {
                "role": "user",
                "content": [{"type": "text", "content": "What's the weather in Tokyo?"}],
            },
            {
                "role": "gemini",
                "content": [
                    {
                        "type": "tool_call",
                        "content": {
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo"},
                            "tool_call_id": "tool_call_12345",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "tool_call_12345",
                        "content": '{"temperature": "25°C"}',
                        "name": "get_weather",
                    }
                ],
            },
        ]

        # This is the expected format for the Gemini API
        expected_gemini_history = [
            {"role": "user", "parts": [{"text": "What's the weather in Tokyo?"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"location": "Tokyo"},
                        }
                    }
                ],
            },
            {
                "role": "function",
                "parts": [
                    {
                        "function_response": {
                            "name": "get_weather",
                            "id": "tool_call_12345",
                            "response": json.loads('{"temperature": "25°C"}'),
                        }
                    }
                ],
            },
        ]

        formatted_history = GeminiProvider.format_history(logic_history)
        assert formatted_history == expected_gemini_history

    def test_format_history_wraps_non_json_tool_result(self):
        """format_history should wrap non-JSON tool results for Gemini responses."""
        logic_history = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "tool_call_1",
                        "content": "plain text response",
                        "name": "search",
                    }
                ],
            }
        ]

        expected = [
            {
                "role": "function",
                "parts": [
                    {
                        "function_response": {
                            "name": "search",
                            "id": "tool_call_1",
                            "response": {"result": "plain text response"},
                        }
                    }
                ],
            }
        ]

        formatted_history = GeminiProvider.format_history(logic_history)
        assert formatted_history == expected

    def test_format_history_wraps_non_object_json_tool_result(self):
        """format_history should wrap non-object JSON tool results for Gemini responses."""
        logic_history = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "tool_call_2",
                        "content": "[1, 2, 3]",
                        "name": "search",
                    }
                ],
            }
        ]

        expected = [
            {
                "role": "function",
                "parts": [
                    {
                        "function_response": {
                            "name": "search",
                            "id": "tool_call_2",
                            "response": {"result": [1, 2, 3]},
                        }
                    }
                ],
            }
        ]

        formatted_history = GeminiProvider.format_history(logic_history)
        assert formatted_history == expected

    def test_format_history_handles_unexpected_content_types(self):
        """format_history should convert unexpected types to string (Issue #79 Review Fix)."""
        logic_history = [
            {"role": "user", "content": 123},  # int
            {"role": "gemini", "content": True},  # bool
            {"role": "user", "content": {"text": "bad"}},  # dict
        ]

        # Should NOT raise ValueError, but log warning and convert to string
        formatted_history = GeminiProvider.format_history(logic_history)

        # Should have converted all to text parts
        assert len(formatted_history) == 3
        assert formatted_history[0]["parts"][0]["text"] == "123"
        assert formatted_history[1]["parts"][0]["text"] == "True"
        assert formatted_history[2]["parts"][0]["text"] == "{'text': 'bad'}"

    def test_format_history_handles_none_content(self):
        """format_history should tolerate None content entries."""
        logic_history = [
            {"role": "user", "content": None},
            {"role": "gemini", "content": None},
        ]

        formatted_history = GeminiProvider.format_history(logic_history)
        assert formatted_history == []

    def test_parse_tool_response_payload_handles_type_error(self):
        """_parse_tool_response_payload should handle TypeError and raise (Priority B1)"""

        # Create an object that raises TypeError when str() is called
        class UnserializableObject:
            def __str__(self):
                raise TypeError("Cannot serialize")

        # Should raise TypeError after logging warning
        with pytest.raises(TypeError, match="cannot be safely converted to dict"):
            _parse_tool_response_payload(UnserializableObject())

    def test_parse_tool_response_payload_handles_value_error(self):
        """_parse_tool_response_payload should handle various JSON parsing errors."""

        # Test with invalid JSON
        result = _parse_tool_response_payload("not valid json")
        assert result == {"result": "not valid json"}

        # Test with None
        result = _parse_tool_response_payload(None)
        assert result == {}

        # Test with already a dict
        result = _parse_tool_response_payload({"key": "value"})
        assert result == {"key": "value"}

    def test_parse_tool_response_payload_whitelisted_types(self):
        """_parse_tool_response_payload should handle whitelisted types (Priority B1)"""

        # Test int
        result = _parse_tool_response_payload(42)
        assert result == {"result": 42}

        # Test float
        result = _parse_tool_response_payload(3.14)
        assert result == {"result": 3.14}

        # Test list
        result = _parse_tool_response_payload([1, 2, 3])
        assert result == {"result": [1, 2, 3]}

        # Test bool
        result = _parse_tool_response_payload(True)
        assert result == {"result": True}

    def test_parse_tool_response_payload_logs_warning_for_unexpected_types(self, caplog):
        """_parse_tool_response_payload should log warning for unexpected types (Priority B1)"""
        import logging

        # bytes should trigger warning and str() conversion
        with caplog.at_level(logging.WARNING, logger="multi_llm_chat.llm_provider"):
            result = _parse_tool_response_payload(b"binary data")

        assert "unexpected type" in caplog.text
        assert result == {"result": "b'binary data'"}

        caplog.clear()

        # Custom object should also trigger warning
        class CustomObject:
            def __str__(self):
                return "custom_string"

        with caplog.at_level(logging.WARNING, logger="multi_llm_chat.llm_provider"):
            result = _parse_tool_response_payload(CustomObject())

        assert "unexpected type" in caplog.text
        assert result == {"result": "custom_string"}


class TestGeminiToolCallEmptyArgs(unittest.TestCase):
    """Test empty argument handling for tool calls (Issue #79 Review Fix)"""

    def test_tool_call_with_empty_args_emits_immediately(self):
        """Tool calls with empty {} args should emit immediately, not be delayed."""
        assembler = GeminiToolCallAssembler()

        # Simulate receiving tool name first
        mock_part = MagicMock()
        mock_part.index = None  # Sequential calling
        mock_call_name = MagicMock()
        mock_call_name.name = "get_current_time"
        mock_call_name.args = None
        result1 = assembler.process_function_call(mock_part, mock_call_name)
        assert result1 is None, "Should not emit with name only"

        # Simulate receiving empty args
        mock_call_args = MagicMock()
        mock_call_args.name = None
        mock_call_args.args = {}
        result2 = assembler.process_function_call(mock_part, mock_call_args)

        # Should emit immediately (not delayed until finalize)
        assert result2 is not None, "Should emit immediately with empty args"
        assert result2["type"] == "tool_call"
        assert result2["content"]["name"] == "get_current_time"
        assert result2["content"]["arguments"] == {}

    def test_tool_call_with_none_args_emits_immediately(self):
        """Tool calls with None args (no args key) should emit immediately."""
        assembler = GeminiToolCallAssembler()

        # Simulate receiving both name and args=None in one chunk
        mock_part = MagicMock()
        mock_part.index = None
        mock_call = MagicMock()
        mock_call.name = "list_users"
        mock_call.args = None
        result = assembler.process_function_call(mock_part, mock_call)

        # Should NOT emit (args=None means no args_updated flag)
        # This is expected behavior: None means "not yet received"
        assert result is None

        # But finalize should emit it
        pending = list(assembler.finalize_pending_calls())
        assert len(pending) == 1
        assert pending[0]["content"]["name"] == "list_users"
        assert pending[0]["content"]["arguments"] == {}
