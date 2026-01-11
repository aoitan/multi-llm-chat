"""
Tests for Gemini Provider tool integration.
"""

import unittest
from unittest.mock import Mock, patch

from multi_llm_chat.llm_provider import GeminiProvider


class TestGeminiToolsIntegration(unittest.TestCase):
    """Test GeminiProvider with MCP tools integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.provider = GeminiProvider()

        # Sample MCP tool definition
        self.mcp_tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]

    def test_mcp_to_gemini_tool_conversion(self):
        """MCP形式のツール定義をGemini形式に変換できる"""
        from multi_llm_chat.llm_provider import mcp_tools_to_gemini_format

        gemini_tools = mcp_tools_to_gemini_format(self.mcp_tools)

        # Gemini形式の検証
        self.assertIsInstance(gemini_tools, list)
        self.assertEqual(len(gemini_tools), 1)

        # Tool構造の検証
        tool = gemini_tools[0]
        self.assertIn("function_declarations", tool)

        func_decl = tool["function_declarations"][0]
        self.assertEqual(func_decl["name"], "get_weather")
        self.assertEqual(func_decl["description"], "Get current weather for a location")
        self.assertIn("parameters", func_decl)

    @patch("multi_llm_chat.llm_provider.genai.GenerativeModel")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test_key")
    def test_gemini_provider_call_api_with_tools(self, mock_model_class):
        """GeminiProvider.call_apiにtools引数を渡せる"""
        # Setup mock
        mock_model = Mock()
        mock_response = Mock()
        mock_response.__iter__ = Mock(return_value=iter([Mock(text="Response")]))
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        history = [{"role": "user", "content": "What's the weather?"}]

        # Call with tools
        list(self.provider.call_api(history, tools=self.mcp_tools))

        # Verify model.generate_content was called with tools
        self.assertEqual(mock_model.generate_content.call_count, 1)
        call_args = mock_model.generate_content.call_args

        # Check that tools were passed (keyword argument)
        self.assertIn("tools", call_args.kwargs)

    def test_parse_gemini_function_call(self):
        """Gemini FunctionCallを共通形式にパースできる"""
        from multi_llm_chat.llm_provider import parse_gemini_function_call

        # Mock Gemini FunctionCall
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "Tokyo", "unit": "celsius"}

        result = parse_gemini_function_call(mock_function_call)

        # 共通形式の検証
        self.assertEqual(result["tool_name"], "get_weather")
        self.assertEqual(result["arguments"], {"location": "Tokyo", "unit": "celsius"})

    @patch("multi_llm_chat.llm_provider.genai.GenerativeModel")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test_key")
    def test_gemini_response_with_function_call(self, mock_model_class):
        """Geminiからのツール呼び出しレスポンスを処理できる"""
        # Setup mock with function call
        mock_model = Mock()
        mock_chunk = Mock()
        mock_chunk.text = ""

        # Mock function call in chunk
        mock_part = Mock()
        mock_part.function_call.name = "get_weather"
        mock_part.function_call.args = {"location": "Tokyo"}
        mock_chunk.parts = [mock_part]

        mock_response = Mock()
        mock_response.__iter__ = Mock(return_value=iter([mock_chunk]))
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        history = [{"role": "user", "content": "What's the weather in Tokyo?"}]

        # Call API
        chunks = list(self.provider.call_api(history, tools=self.mcp_tools))

        # Verify we got chunks
        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
