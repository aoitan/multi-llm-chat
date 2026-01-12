import unittest
from unittest.mock import MagicMock, patch

from google.generativeai.types import Tool

from multi_llm_chat.llm_provider import (
    GeminiProvider,
    mcp_tools_to_gemini_format,
)

"""Tests for Gemini Tools integration (Issue #79)

TDD Development Notes (2026-01-12):
    This test suite was developed following the Red-Green-Refactor TDD cycle
    to implement parallel function calling support for GeminiProvider.

    RED Phase:
        - test_call_api_handles_interleaved_parallel_tool_calls (added to test_llm_provider.py)
          → FAILED due to FIFO index assignment causing argument mismatch
    
    GREEN Phase:
        - Refactored GeminiToolCallAssembler to use hybrid index/FIFO tracking
        - Indexed parts (parallel calls): Dictionary-based tracking by part.index
        - Non-indexed parts (sequential calls): FIFO queue strategy
        → ALL TESTS PASSED
    
    REFACTOR Phase:
        - Removed dead code (_last_key variable)
        - Added detailed docstrings explaining state management
        - Added logging for debugging parallel tool call assembly
    
    Critical Fix:
        The original FIFO-only implementation failed when Gemini API sent
        parallel tool calls with interleaved argument chunks. The hybrid
        approach correctly matches arguments to their corresponding tool names
        using the part.index attribute when available.
"""


class TestGeminiToolsIntegration(unittest.TestCase):
    """MCPツールとGemini Providerの連携をテスト"""

    def setUp(self):
        """テスト用のProviderとツール定義を準備"""
        self.provider = GeminiProvider()
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
        self.history = [{"role": "user", "content": "What's the weather in Tokyo?"}]

    def test_mcp_tools_to_gemini_format(self):
        """MCP形式からGemini Tools形式への変換が正しく行われること"""
        gemini_tools = mcp_tools_to_gemini_format(self.mcp_tools)
        self.assertIsInstance(gemini_tools, list)
        self.assertEqual(len(gemini_tools), 1)
        self.assertIsInstance(gemini_tools[0], Tool)

        declarations = gemini_tools[0].function_declarations
        self.assertIsNotNone(declarations)
        self.assertEqual(len(declarations), 1)

        func_decl = declarations[0]
        self.assertEqual(func_decl.name, "get_weather")
        self.assertEqual(func_decl.description, self.mcp_tools[0]["description"])

        # Instead of comparing dicts, check the attributes of the Schema object
        schema_obj = func_decl.parameters
        expected_schema_dict = self.mcp_tools[0]["inputSchema"]

        self.assertEqual(
            schema_obj.properties["location"].description,
            expected_schema_dict["properties"]["location"]["description"],
        )
        self.assertEqual(list(schema_obj.required), expected_schema_dict["required"])

    @patch("multi_llm_chat.llm_provider.genai.GenerativeModel")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test_key")
    def test_gemini_response_with_function_call(self, mock_model_class):
        """Geminiからのツール呼び出しレスポンスを共通形式にパースして返す"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.generate_content.return_value = mock_response

        # Mock stream for text followed by a tool call
        mock_text_chunk = MagicMock(text="Some text", parts=[])

        # --- Tool Call Part (name first, then args in separate chunks) ---
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None  # Name chunk has no args yet
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1
        mock_fc_chunk1 = MagicMock(text=None, parts=[part1])

        fc2 = MagicMock()
        fc2.name = None  # Args chunk has no name
        fc2.args = {"location": "Tokyo"}
        part2 = MagicMock(spec=["function_call"])
        part2.function_call = fc2
        mock_fc_chunk2 = MagicMock(text=None, parts=[part2])
        # --- End Tool Call Part ---

        mock_response.__iter__.return_value = iter(
            [mock_text_chunk, mock_fc_chunk1, mock_fc_chunk2]
        )

        chunks = list(self.provider.call_api(self.history, tools=self.mcp_tools))

        self.assertEqual(len(chunks), 2)

        text_chunk = next((c for c in chunks if c["type"] == "text"), None)
        tool_chunk = next((c for c in chunks if c["type"] == "tool_call"), None)

        self.assertIsNotNone(text_chunk)
        self.assertEqual(text_chunk["content"], "Some text")

        self.assertIsNotNone(tool_chunk)
        self.assertEqual(tool_chunk["content"]["name"], "get_weather")
        self.assertEqual(tool_chunk["content"]["arguments"], {"location": "Tokyo"})

    @patch("multi_llm_chat.llm_provider.genai.GenerativeModel")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test_key")
    def test_gemini_tool_call_order_is_preserved(self, mock_model_class):
        """ツール呼び出しがテキストより先に来た場合の順序を保証する"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.generate_content.return_value = mock_response

        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None  # Name chunk only
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1
        mock_fc_chunk1 = MagicMock(text=None, parts=[part1])

        fc2 = MagicMock()
        fc2.name = None  # Args chunk only
        fc2.args = {"location": "Osaka"}
        part2 = MagicMock(spec=["function_call"])
        part2.function_call = fc2
        mock_fc_chunk2 = MagicMock(text=None, parts=[part2])

        mock_text_chunk = MagicMock(text="Done", parts=[])

        mock_response.__iter__.return_value = iter(
            [mock_fc_chunk1, mock_fc_chunk2, mock_text_chunk]
        )

        chunks = list(self.provider.call_api(self.history, tools=self.mcp_tools))

        self.assertEqual([chunk["type"] for chunk in chunks], ["tool_call", "text"])
        self.assertEqual(chunks[0]["content"]["name"], "get_weather")
        self.assertEqual(chunks[0]["content"]["arguments"], {"location": "Osaka"})
        self.assertEqual(chunks[1]["content"], "Done")

    @patch("multi_llm_chat.llm_provider.genai.GenerativeModel")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test_key")
    def test_gemini_tool_call_args_after_text_are_preserved(self, mock_model_class):
        """ツール呼び出しの引数がテキスト後に届いても欠落しないこと"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.generate_content.return_value = mock_response

        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None  # Name chunk only
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1
        mock_fc_chunk1 = MagicMock(text=None, parts=[part1])

        mock_text_chunk = MagicMock(text="Streaming text", parts=[])

        fc2 = MagicMock()
        fc2.name = None  # Args chunk only
        fc2.args = {"location": "Nagoya"}
        part2 = MagicMock(spec=["function_call"])
        part2.function_call = fc2
        mock_fc_chunk2 = MagicMock(text=None, parts=[part2])

        mock_response.__iter__.return_value = iter(
            [mock_fc_chunk1, mock_text_chunk, mock_fc_chunk2]
        )

        chunks = list(self.provider.call_api(self.history, tools=self.mcp_tools))

        self.assertEqual([chunk["type"] for chunk in chunks], ["text", "tool_call"])
        self.assertEqual(chunks[0]["content"], "Streaming text")
        self.assertEqual(chunks[1]["content"]["name"], "get_weather")
        self.assertEqual(chunks[1]["content"]["arguments"], {"location": "Nagoya"})

    @patch("multi_llm_chat.llm_provider.genai.GenerativeModel")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test_key")
    def test_no_tool_calls_emitted_on_blocked_prompt(self, mock_model_class):
        """BlockedPromptExceptionが発生した場合、未完のツール呼び出しを出力しない"""
        import google.generativeai as genai

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Simulate BlockedPromptException immediately
        mock_model.generate_content.side_effect = genai.types.BlockedPromptException(
            "Safety filter triggered"
        )

        with self.assertRaises(ValueError) as context:
            list(self.provider.call_api(self.history, tools=self.mcp_tools))

        self.assertIn("blocked", str(context.exception).lower())

    @patch("multi_llm_chat.llm_provider.genai.GenerativeModel")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test_key")
    def test_no_tool_calls_emitted_on_api_error(self, mock_model_class):
        """API エラーが発生した場合、未完のツール呼び出しを出力しない"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.generate_content.return_value = mock_response

        # Function call name chunk only (no args)
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1
        mock_fc_chunk1 = MagicMock(text=None, parts=[part1])

        # Simulate generic exception after partial tool call
        def raise_exception():
            yield mock_fc_chunk1
            raise Exception("API connection error")

        mock_response.__iter__.return_value = raise_exception()

        with self.assertRaises(Exception) as context:
            list(self.provider.call_api(self.history, tools=self.mcp_tools))

        self.assertIn("API connection error", str(context.exception))

    @patch("multi_llm_chat.llm_provider.genai.GenerativeModel")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test_key")
    def test_tool_calls_emitted_only_on_success(self, mock_model_class):
        """ストリームが正常に完了した場合のみ、未完のツール呼び出しを出力する"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.generate_content.return_value = mock_response

        # Function call name chunk
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None
        part1 = MagicMock(spec=["function_call"])
        part1.function_call = fc1
        mock_fc_chunk1 = MagicMock(text=None, parts=[part1])

        # Function call args chunk
        fc2 = MagicMock()
        fc2.name = None
        fc2.args = {"location": "Tokyo"}
        part2 = MagicMock(spec=["function_call"])
        part2.function_call = fc2
        mock_fc_chunk2 = MagicMock(text=None, parts=[part2])

        # Normal successful stream
        mock_response.__iter__.return_value = iter([mock_fc_chunk1, mock_fc_chunk2])

        chunks = list(self.provider.call_api(self.history, tools=self.mcp_tools))

        # Should emit the complete tool call
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "tool_call")
        self.assertEqual(chunks[0]["content"]["name"], "get_weather")
        self.assertEqual(chunks[0]["content"]["arguments"], {"location": "Tokyo"})
