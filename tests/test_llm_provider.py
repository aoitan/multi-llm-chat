"""Tests for LLM Provider abstraction (Strategy pattern)

Common tests for provider factory, caching, and utilities.
Provider-specific tests are in test_llm_provider_gemini.py and test_llm_provider_openai.py.
"""

import unittest

import pytest

from multi_llm_chat.llm_provider import (
    ChatGPTProvider,
    GeminiProvider,
    LLMProvider,
    _parse_tool_response_payload,
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

    def format_history(self, history):
        return history


def test_stream_text_events_filters_empty_strings():
    """stream_text_events should skip empty strings from chunk extraction."""
    provider = DummyProvider(["Hello", "", "world"])

    result = list(provider.stream_text_events([], None))

    assert result == ["Hello", "world"]


class TestParseToolResponsePayload:
    """Test JSON error handling in tool response parsing (Issue #79 Review Fix)"""

    def test_parse_invalid_json_catches_json_decode_error(self):
        """Invalid JSON strings should be wrapped as {result: ...}"""
        result = _parse_tool_response_payload('{"invalid": json}')
        assert result == {"result": '{"invalid": json}'}

    def test_parse_invalid_json_catches_value_error(self):
        """Extremely nested/large JSON should be handled gracefully."""
        # Simulate a string that triggers ValueError (e.g., extreme recursion)
        # In practice, json.loads may raise ValueError for certain edge cases
        invalid_json = '{"a":' * 1000 + "1" + "}" * 1000  # Deeply nested
        result = _parse_tool_response_payload(invalid_json)
        # Should either parse successfully or wrap as {"result": ...}
        assert isinstance(result, dict)
        assert "result" in result or "a" in result

    def test_parse_valid_json_dict(self):
        """Valid JSON dict should be parsed correctly."""
        result = _parse_tool_response_payload('{"status": "success"}')
        assert result == {"status": "success"}

    def test_parse_valid_json_non_dict(self):
        """Valid JSON non-dict should be wrapped as {result: ...}"""
        result = _parse_tool_response_payload('["a", "b"]')
        assert result == {"result": ["a", "b"]}

    def test_parse_none_payload(self):
        """None payload should return empty dict."""
        result = _parse_tool_response_payload(None)
        assert result == {}

    def test_parse_dict_payload(self):
        """Dict payload should be returned as-is."""
        result = _parse_tool_response_payload({"key": "value"})
        assert result == {"key": "value"}

    def test_parse_primitive_types(self):
        """Primitive types should be wrapped as {result: ...}"""
        assert _parse_tool_response_payload(123) == {"result": 123}
        assert _parse_tool_response_payload(True) == {"result": True}
        assert _parse_tool_response_payload([1, 2, 3]) == {"result": [1, 2, 3]}


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in llm_provider module"""

    def test_load_api_key_reads_from_env(self):
        """load_api_key should read API key from environment"""
        from unittest.mock import patch

        from multi_llm_chat.llm_provider import load_api_key

        with patch("os.getenv", return_value="test_key_12345"):
            key = load_api_key("GOOGLE_API_KEY")
            assert key == "test_key_12345"
