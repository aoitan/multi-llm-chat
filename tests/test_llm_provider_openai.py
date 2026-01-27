"""Tests for ChatGPTProvider (OpenAI) implementation

Extracted from test_llm_provider.py as part of Issue #101 refactoring.
"""

import json
from unittest.mock import MagicMock, patch

from multi_llm_chat.llm_provider import ChatGPTProvider


class TestChatGPTProvider:
    """Test ChatGPTProvider implementation"""

    @patch("multi_llm_chat.llm_provider.OPENAI_API_KEY", "test-key")
    @patch("openai.OpenAI")
    def test_call_api_basic(self, mock_openai_class):
        """ChatGPTProvider.call_api should return a generator"""
        # Setup mock
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([MagicMock()]))
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai_class.return_value = mock_client

        # Test
        provider = ChatGPTProvider()
        history = [{"role": "user", "content": "Hello"}]
        result = provider.call_api(history)

        # Verify it's a generator
        assert hasattr(result, "__iter__")
        list(result)  # Consume generator
        mock_client.chat.completions.create.assert_called_once()

    @patch("multi_llm_chat.llm_provider.OPENAI_API_KEY", "test-key")
    @patch("openai.OpenAI")
    def test_call_api_yields_text_chunks(self, mock_openai_class):
        """ChatGPTProvider.call_api should yield unified text chunks."""
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])
        mock_openai_class.return_value = mock_client

        provider = ChatGPTProvider()
        history = [{"role": "user", "content": "Hello"}]
        result = list(provider.call_api(history))

        assert result == [
            {"type": "text", "content": "Hello"},
            {"type": "text", "content": " world"},
        ]

    @patch("multi_llm_chat.llm_provider.OPENAI_API_KEY", "test-key")
    @patch("openai.OpenAI")
    def test_call_api_with_tools(self, mock_openai_class):
        """ChatGPTProvider.call_api should accept tools parameter (Issue #80)."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock empty stream
        mock_stream = iter([])
        mock_client.chat.completions.create.return_value = mock_stream

        provider = ChatGPTProvider()
        history = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test_tool", "description": "Test", "inputSchema": {}}]

        # Should not raise - tools are now supported
        list(provider.call_api(history, tools=tools))

        # Verify API was called with tools
        call_args = mock_client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs
        assert "tool_choice" in call_args.kwargs

    def test_extract_text_from_chunk(self):
        """ChatGPTProvider should extract text from response chunk"""
        provider = ChatGPTProvider()
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hello, world!"

        result = provider.extract_text_from_chunk(chunk)
        assert result == "Hello, world!"

    def test_extract_text_from_chunk_list_format(self):
        """ChatGPTProvider should handle list format in delta.content"""
        provider = ChatGPTProvider()
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        # Simulate list format response (e.g., from function calls)
        chunk.choices[0].delta.content = ["Part 1", "Part 2"]

        result = provider.extract_text_from_chunk(chunk)
        # Should join list elements without space to preserve JSON structure
        assert result == "Part 1Part 2"

    def test_extract_text_from_chunk_dict_format(self):
        """ChatGPTProvider should handle dictionary chunk format."""
        provider = ChatGPTProvider()
        chunk = {"choices": [{"delta": {"content": "Hello from dict"}}]}

        result = provider.extract_text_from_chunk(chunk)
        assert result == "Hello from dict"

    def test_extract_text_from_chunk_none_content(self):
        """ChatGPTProvider should return empty string for None content"""
        provider = ChatGPTProvider()
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None

        result = provider.extract_text_from_chunk(chunk)
        assert result == ""

    def test_get_token_info_returns_dict(self):
        """ChatGPTProvider should return token info dictionary"""
        provider = ChatGPTProvider()
        result = provider.get_token_info("Test message")

        assert isinstance(result, dict)
        assert "input_tokens" in result
        assert "max_tokens" in result

    def test_format_history_with_tool_role(self):
        """ChatGPTProvider.format_history should handle role='tool' entries from Agentic Loop"""
        logic_history = [
            {
                "role": "user",
                "content": [{"type": "text", "content": "Get weather"}],
            },
            {
                "role": "chatgpt",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_123",
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_123",
                        "name": "get_weather",
                        "content": "25°C",
                    }
                ],
            },
        ]

        expected_chatgpt_history = [
            {"role": "user", "content": "Get weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Tokyo"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "25°C",
            },
        ]

        formatted_history = ChatGPTProvider.format_history(logic_history)
        assert formatted_history == expected_chatgpt_history
