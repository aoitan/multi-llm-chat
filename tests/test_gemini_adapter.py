"""Tests for Gemini SDK Adapter layer

Issue: #136 (Phase 1 - Adapter層導入)
Issue: #138 (Phase 3 - New SDK as Default)
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from multi_llm_chat.providers.gemini_adapter import GeminiSDKAdapter, LegacyGeminiAdapter


class TestGeminiSDKAdapter(unittest.TestCase):
    """Test abstract base class"""

    def test_is_abstract(self):
        """GeminiSDKAdapter should be abstract"""
        with self.assertRaises(TypeError):
            GeminiSDKAdapter()


# Skip legacy adapter tests when new SDK is used (default after Issue #138)
@pytest.mark.skipif(
    os.getenv("USE_LEGACY_GEMINI_SDK", "0") == "0",
    reason="Legacy adapter tests only run with USE_LEGACY_GEMINI_SDK=1",
)
class TestLegacyGeminiAdapter(unittest.TestCase):
    """Test LegacyGeminiAdapter wrapper"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_genai = MagicMock()
        self.mock_model = MagicMock()

    @patch("google.generativeai")
    def test_init_configures_api_key(self, mock_genai):
        """Should configure genai with API key"""
        adapter = LegacyGeminiAdapter(api_key="test_key")

        mock_genai.configure.assert_called_once_with(api_key="test_key")
        self.assertIsNotNone(adapter)

    @patch("google.generativeai")
    def test_get_model_without_system_instruction(self, mock_genai):
        """Should cache default model without system instruction"""
        mock_genai.GenerativeModel.return_value = self.mock_model

        adapter = LegacyGeminiAdapter(api_key="test_key")
        model1 = adapter.get_model("gemini-pro")
        model2 = adapter.get_model("gemini-pro")

        # Should call GenerativeModel once (cached)
        self.assertEqual(mock_genai.GenerativeModel.call_count, 1)
        self.assertIs(model1, model2)

    @patch("google.generativeai")
    def test_get_model_with_system_instruction(self, mock_genai):
        """Should cache models with system instructions separately"""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_genai.GenerativeModel.side_effect = [mock_model1, mock_model2]

        adapter = LegacyGeminiAdapter(api_key="test_key")
        model1 = adapter.get_model("gemini-pro", "You are a helpful assistant")
        model2 = adapter.get_model("gemini-pro", "You are a code reviewer")

        # Should call GenerativeModel twice (different prompts)
        self.assertEqual(mock_genai.GenerativeModel.call_count, 2)
        self.assertIsNot(model1, model2)

    @patch("google.generativeai")
    def test_get_model_caches_same_system_instruction(self, mock_genai):
        """Should cache same system instruction"""
        mock_genai.GenerativeModel.return_value = self.mock_model

        adapter = LegacyGeminiAdapter(api_key="test_key")
        model1 = adapter.get_model("gemini-pro", "You are helpful")
        model2 = adapter.get_model("gemini-pro", "You are helpful")

        # Should call GenerativeModel once (cached)
        self.assertEqual(mock_genai.GenerativeModel.call_count, 1)
        self.assertIs(model1, model2)

    @patch("google.generativeai")
    def test_lru_eviction(self, mock_genai):
        """Should evict least recently used models when cache is full"""
        mock_genai.GenerativeModel.side_effect = [MagicMock() for _ in range(12)]

        adapter = LegacyGeminiAdapter(api_key="test_key")
        adapter._cache_max_size = 10

        # Fill cache with 11 models
        for i in range(11):
            adapter.get_model("gemini-pro", f"prompt_{i}")

        # Cache should have max 10 items
        self.assertLessEqual(len(adapter._models_cache), 10)

    @patch("google.generativeai")
    def test_different_model_names(self, mock_genai):
        """Should cache models separately by model name"""
        model1 = MagicMock()
        model2 = MagicMock()
        mock_genai.GenerativeModel.side_effect = [model1, model2]

        adapter = LegacyGeminiAdapter(api_key="test_key")

        # Get two different models with same system instruction
        result1 = adapter.get_model("gemini-1.5-pro", "same prompt")
        result2 = adapter.get_model("gemini-2.0-flash", "same prompt")

        # Should return different model instances
        self.assertIs(result1, model1)
        self.assertIs(result2, model2)
        self.assertIsNot(result1, result2)

        # Should have 2 cache entries with different keys
        self.assertEqual(len(adapter._models_cache), 2)

    @patch("google.generativeai")
    def test_default_model_by_name(self, mock_genai):
        """Should cache default models separately by model name"""
        model1 = MagicMock()
        model2 = MagicMock()
        mock_genai.GenerativeModel.side_effect = [model1, model2]

        adapter = LegacyGeminiAdapter(api_key="test_key")

        # Get two different models without system instruction
        result1 = adapter.get_model("gemini-1.5-pro")
        result2 = adapter.get_model("gemini-2.0-flash")

        # Should return different model instances
        self.assertIs(result1, model1)
        self.assertIs(result2, model2)

        # Should have 2 default model entries
        self.assertEqual(len(adapter._default_models), 2)
        self.assertIn("gemini-1.5-pro", adapter._default_models)
        self.assertIn("gemini-2.0-flash", adapter._default_models)

    @patch("google.generativeai")
    def test_hash_prompt(self, mock_genai):
        """Should generate consistent hash for prompts"""
        adapter = LegacyGeminiAdapter(api_key="test_key")

        hash1 = adapter._hash_prompt("test prompt")
        hash2 = adapter._hash_prompt("test prompt")
        hash3 = adapter._hash_prompt("different prompt")

        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertEqual(len(hash1), 64)  # SHA256 hex length

    @patch("google.generativeai")
    def test_handle_api_errors_blocked_prompt(self, mock_genai):
        """Should convert BlockedPromptException to ValueError"""

        # Create BlockedPromptException class
        class BlockedPromptException(Exception):
            pass

        mock_genai.types.BlockedPromptException = BlockedPromptException
        adapter = LegacyGeminiAdapter(api_key="test_key")

        # Test that BlockedPromptException is converted to ValueError
        with self.assertRaises(ValueError) as ctx:
            with adapter.handle_api_errors():
                raise BlockedPromptException("Safety filter triggered")

        self.assertIn("Prompt was blocked", str(ctx.exception))
        self.assertIn("Safety filter triggered", str(ctx.exception))

    @patch("google.generativeai")
    def test_handle_api_errors_passthrough(self, mock_genai):
        """Should pass through other exceptions unchanged"""
        adapter = LegacyGeminiAdapter(api_key="test_key")

        # Test that other exceptions pass through
        with self.assertRaises(RuntimeError) as ctx:
            with adapter.handle_api_errors():
                raise RuntimeError("Some other error")

        self.assertEqual(str(ctx.exception), "Some other error")


if __name__ == "__main__":
    unittest.main()
