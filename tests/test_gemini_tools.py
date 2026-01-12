import unittest
from unittest.mock import MagicMock, patch

from google.generativeai.types import Tool

from multi_llm_chat.llm_provider import (
    GeminiProvider,
    mcp_tools_to_gemini_format,
)


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

        # --- Tool Call Part ---
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = {}  # Gemini can send an empty args dict
        part1 = MagicMock()
        part1.function_call = fc1
        mock_fc_chunk1 = MagicMock(text=None, parts=[part1])

        fc2 = MagicMock()
        fc2.name = None
        fc2.args = {"location": "Tokyo"}
        part2 = MagicMock()
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
