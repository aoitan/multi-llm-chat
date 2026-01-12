"""Tests for LLM Provider abstraction (Strategy pattern)"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from multi_llm_chat.llm_provider import (
    ChatGPTProvider,
    GeminiProvider,
    LLMProvider,
    create_provider,
    get_provider,
)


class TestLLMProviderFactory(unittest.TestCase):
    """Test the provider factory functions"""

    def test_create_provider_gemini(self):
        """create_provider should return new GeminiProvider for 'gemini'"""
        provider = create_provider("gemini")
        assert isinstance(provider, GeminiProvider)

    def test_create_provider_chatgpt(self):
        """create_provider should return new ChatGPTProvider for 'chatgpt'"""
        provider = create_provider("chatgpt")
        assert isinstance(provider, ChatGPTProvider)

    def test_create_provider_invalid(self):
        """create_provider should raise ValueError for unknown provider"""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider("unknown")

    def test_create_provider_returns_new_instances(self):
        """create_provider should return new instance each time"""
        provider1 = create_provider("gemini")
        provider2 = create_provider("gemini")
        # Should return different instances
        assert provider1 is not provider2

    def test_get_provider_gemini(self):
        """get_provider should return GeminiProvider for 'gemini' (DEPRECATED)"""
        provider = get_provider("gemini")
        assert isinstance(provider, GeminiProvider)

    def test_get_provider_chatgpt(self):
        """get_provider should return ChatGPTProvider for 'chatgpt' (DEPRECATED)"""
        provider = get_provider("chatgpt")
        assert isinstance(provider, ChatGPTProvider)

    def test_get_provider_invalid(self):
        """get_provider should raise ValueError for unknown provider (DEPRECATED)"""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("unknown")


class TestGeminiProvider(unittest.TestCase):
    """Test GeminiProvider implementation"""

    def setUp(self):
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
        self.assertEqual(result, expected)
        mock_model.generate_content.assert_called_once()

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_yields_single_tool_call(self, mock_genai):
        """call_api should yield a unified tool_call dictionary for a single tool call."""
        # Mock the Gemini API response for a tool call
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None
        part1 = MagicMock()
        part1.function_call = fc1

        fc2 = MagicMock()
        fc2.name = None
        fc2.args = {"location": "Tokyo"}
        part2 = MagicMock()
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
        self.assertEqual(result, expected)

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_yields_multiple_tool_calls(self, mock_genai):
        """call_api should yield multiple tool_call dictionaries."""
        # Mock stream for two separate tool calls
        fc1 = MagicMock()
        fc1.name = "get_weather"
        fc1.args = None
        part1 = MagicMock()
        part1.function_call = fc1

        fc2 = MagicMock()
        fc2.name = None
        fc2.args = {"location": "Tokyo"}
        part2 = MagicMock()
        part2.function_call = fc2

        fc3 = MagicMock()
        fc3.name = "get_time"
        fc3.args = None
        part3 = MagicMock()
        part3.function_call = fc3

        fc4 = MagicMock()
        fc4.name = None
        fc4.args = {"timezone": "JST"}
        part4 = MagicMock()
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
        self.assertEqual(result, expected)

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_handles_mixed_text_and_tool_call(self, mock_genai):
        """call_api should handle responses with both text and tool calls."""
        text_part = MagicMock(text="Thinking about it...", parts=[])

        fc1 = MagicMock()
        fc1.name = "search"
        fc1.args = None
        tool_part1 = MagicMock()
        tool_part1.function_call = fc1

        fc2 = MagicMock()
        fc2.name = None
        fc2.args = {"query": "python"}
        tool_part2 = MagicMock()
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
        self.assertEqual(result, expected)

    def test_extract_text_from_chunk(self):
        """extract_text_from_chunk should process unified dictionaries."""
        provider = GeminiProvider()

        text_chunk = {"type": "text", "content": "Hello, world!"}
        self.assertEqual(provider.extract_text_from_chunk(text_chunk), "Hello, world!")

        tool_chunk = {"type": "tool_call", "content": {"name": "test", "arguments": {}}}
        self.assertEqual(provider.extract_text_from_chunk(tool_chunk), "")

        invalid_chunk = {"type": "other", "content": "something"}
        self.assertEqual(provider.extract_text_from_chunk(invalid_chunk), "")

        # Test backward compatibility with old chunk format (should return empty)
        old_chunk = MagicMock(text="some text")
        self.assertEqual(provider.extract_text_from_chunk(old_chunk), "")

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
        provider = GeminiProvider()
        
        # This is the application's internal structured history format
        logic_history = [
            {"role": "user", "content": [{"type": "text", "content": "What's the weather in Tokyo?"}]},
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
                        "name": "get_weather"
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
                            "response": {"content": '{"temperature": "25°C"}'},
                        }
                    }
                ],
            },
        ]
        
        formatted_history = GeminiProvider.format_history(logic_history)
        self.assertEqual(formatted_history, expected_gemini_history)





class TestChatGPTProvider(unittest.TestCase):
    """Test ChatGPTProvider implementation"""

    @patch("multi_llm_chat.llm_provider.OPENAI_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.openai.OpenAI")
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


class TestProviderCaching(unittest.TestCase):
    """Test provider instance caching"""

    def test_get_provider_caches_instances(self):
        """get_provider should return same instance for same provider name"""
        provider1 = get_provider("gemini")
        provider2 = get_provider("gemini")

        # Should return the same cached instance
        assert provider1 is provider2

    def test_get_provider_different_providers(self):
        """get_provider should return different instances for different providers"""
        gemini_provider = get_provider("gemini")
        chatgpt_provider = get_provider("chatgpt")

        # Should return different instances
        assert gemini_provider is not chatgpt_provider
        assert isinstance(gemini_provider, GeminiProvider)
        assert isinstance(chatgpt_provider, ChatGPTProvider)


class DummyProvider(LLMProvider):
    """Minimal provider for testing shared stream behavior."""

    def __init__(self, chunks):
        self._chunks = chunks

    def call_api(self, history, system_prompt=None):
        return iter(self._chunks)

    def extract_text_from_chunk(self, chunk):
        return chunk

    def get_token_info(self, text, history=None, model_name=None):
        return {"input_tokens": 0, "max_tokens": 0}


def test_stream_text_events_filters_empty_strings():
    """stream_text_events should skip empty strings from chunk extraction."""
    provider = DummyProvider(["Hello", "", "world"])

    result = list(provider.stream_text_events([], None))

    assert result == ["Hello", "world"]
