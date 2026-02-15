"""Tests for NewGeminiAdapter (google.genai SDK)

Tests the new SDK adapter implementation using TDD approach.
Issue: #137 (Phase 2: New SDK Implementation)
"""

import unittest
from unittest.mock import Mock, patch

from multi_llm_chat.providers.gemini_client import NewGeminiAdapter


class TestNewGeminiAdapterBasics(unittest.TestCase):
    """Basic initialization and configuration tests"""

    @patch("google.genai.Client")
    def test_init_creates_client(self, mock_client_class):
        """NewGeminiAdapter should create google.genai.Client with api_key"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key="test-key-123")

        mock_client_class.assert_called_once_with(api_key="test-key-123")
        self.assertEqual(adapter.client, mock_client)

    @patch("google.genai.Client")
    def test_init_with_none_api_key(self, mock_client_class):
        """NewGeminiAdapter should handle None api_key gracefully"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key=None)

        # Client created with None (SDK will handle the error later)
        mock_client_class.assert_called_once_with(api_key=None)
        self.assertEqual(adapter.client, mock_client)


class TestGetModel(unittest.TestCase):
    """Tests for get_model() method"""

    @patch("google.genai.Client")
    def test_get_model_returns_model_proxy(self, mock_client_class):
        """get_model should return _ModelProxy object"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key="test-key")

        model = adapter.get_model("gemini-2.0-flash-exp")

        # Returns proxy object (not string)
        self.assertTrue(hasattr(model, "generate_content"))
        self.assertEqual(model._model_name, "gemini-2.0-flash-exp")
        self.assertIsNone(model._system_instruction)

    @patch("google.genai.Client")
    def test_get_model_with_system_instruction(self, mock_client_class):
        """get_model with system_instruction should return proxy with system instruction"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key="test-key")

        model = adapter.get_model("gemini-2.0-flash-exp", system_instruction="You are helpful")

        # Returns proxy with system instruction
        self.assertEqual(model._model_name, "gemini-2.0-flash-exp")
        self.assertEqual(model._system_instruction, "You are helpful")

    @patch("google.genai.Client")
    def test_get_model_caches_by_name_and_system(self, mock_client_class):
        """get_model should cache models by (name, system_instruction) tuple"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key="test-key")

        model1 = adapter.get_model("gemini-2.0-flash-exp")
        model2 = adapter.get_model("gemini-2.0-flash-exp")

        # Same model name + no system returns same proxy instance (cached)
        self.assertIs(model1, model2)

    @patch("google.genai.Client")
    def test_get_model_different_names(self, mock_client_class):
        """get_model should cache different models separately"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key="test-key")

        model_flash = adapter.get_model("gemini-2.0-flash-exp")
        model_pro = adapter.get_model("gemini-1.5-pro")

        # Different model names
        self.assertNotEqual(model_flash, model_pro)


class TestHandleApiErrors(unittest.TestCase):
    """Tests for handle_api_errors() context manager"""

    @patch("google.genai.Client")
    def test_handle_api_errors_no_exception(self, mock_client_class):
        """handle_api_errors should allow normal execution"""
        adapter = NewGeminiAdapter(api_key="test-key")

        with adapter.handle_api_errors():
            result = "success"

        self.assertEqual(result, "success")

    @patch("google.genai.Client")
    def test_handle_api_errors_passthrough(self, mock_client_class):
        """handle_api_errors should pass through non-SDK exceptions"""
        adapter = NewGeminiAdapter(api_key="test-key")

        with self.assertRaises(ValueError) as ctx:
            with adapter.handle_api_errors():
                raise ValueError("Generic error")

        self.assertEqual(str(ctx.exception), "Generic error")


if __name__ == "__main__":
    unittest.main()
