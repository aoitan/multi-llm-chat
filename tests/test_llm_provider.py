"""Tests for LLM Provider abstraction (Strategy pattern)"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from multi_llm_chat.llm_provider import ChatGPTProvider, GeminiProvider, get_provider


class TestLLMProviderFactory(unittest.TestCase):
    """Test the provider factory function"""

    def test_get_provider_gemini(self):
        """get_provider should return GeminiProvider for 'gemini'"""
        provider = get_provider("gemini")
        assert isinstance(provider, GeminiProvider)

    def test_get_provider_chatgpt(self):
        """get_provider should return ChatGPTProvider for 'chatgpt'"""
        provider = get_provider("chatgpt")
        assert isinstance(provider, ChatGPTProvider)

    def test_get_provider_invalid(self):
        """get_provider should raise ValueError for unknown provider"""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("unknown")


class TestGeminiProvider(unittest.TestCase):
    """Test GeminiProvider implementation"""

    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    @patch("multi_llm_chat.llm_provider.genai")
    def test_call_api_basic(self, mock_genai):
        """GeminiProvider.call_api should return a generator"""
        # Setup mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.__iter__ = MagicMock(return_value=iter([MagicMock()]))
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Test
        provider = GeminiProvider()
        history = [{"role": "user", "content": "Hello"}]
        result = provider.call_api(history)

        # Verify it's a generator
        assert hasattr(result, "__iter__")
        list(result)  # Consume generator
        mock_model.generate_content.assert_called_once()

    def test_extract_text_from_chunk(self):
        """GeminiProvider should extract text from response chunk"""
        provider = GeminiProvider()
        chunk = MagicMock()
        chunk.text = "Hello, world!"

        result = provider.extract_text_from_chunk(chunk)
        assert result == "Hello, world!"

    def test_get_token_info_returns_dict(self):
        """GeminiProvider should return token info dictionary"""
        provider = GeminiProvider()
        result = provider.get_token_info("Test message")

        assert isinstance(result, dict)
        assert "input_tokens" in result
        assert "max_tokens" in result


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
