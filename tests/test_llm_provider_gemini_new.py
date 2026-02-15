"""Tests for GeminiProvider with New SDK (google.genai)

These tests verify that GeminiProvider works correctly with the new SDK
when USE_NEW_GEMINI_SDK=1 is set.

Issue: #137 (Phase 2: New SDK Implementation)
"""

import os
import unittest
from unittest.mock import Mock, patch

import pytest

from multi_llm_chat.providers.gemini import GeminiProvider

# Skip these tests unless USE_NEW_GEMINI_SDK=1
pytestmark = pytest.mark.skipif(
    os.getenv("USE_NEW_GEMINI_SDK", "0") != "1",
    reason="These tests only run with USE_NEW_GEMINI_SDK=1",
)


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


if __name__ == "__main__":
    unittest.main()
