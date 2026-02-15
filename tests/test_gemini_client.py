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


class TestCachingAndThreadSafety(unittest.TestCase):
    """Tests for LRU caching and thread safety"""

    @patch("google.genai.Client")
    def test_lru_eviction(self, mock_client_class):
        """LRU cache should evict oldest entries when exceeding max_size"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key="test-key")
        # Override max size for testing
        adapter._cache_max_size = 3

        # Create 4 models (exceeds max_size=3)
        _model1 = adapter.get_model("model1", "prompt1")
        _model2 = adapter.get_model("model2", "prompt2")
        _model3 = adapter.get_model("model3", "prompt3")
        _model4 = adapter.get_model("model4", "prompt4")  # Should evict model1

        # Cache should only have 3 entries
        self.assertEqual(len(adapter._model_cache), 3)

        # model1 should have been evicted
        # Create a new cache key for model1
        prompt_hash = adapter._hash_prompt("prompt1")
        model1_key = ("model1", prompt_hash)
        self.assertNotIn(model1_key, adapter._model_cache)

        # model2, model3, model4 should still be cached
        prompt_hash2 = adapter._hash_prompt("prompt2")
        prompt_hash3 = adapter._hash_prompt("prompt3")
        prompt_hash4 = adapter._hash_prompt("prompt4")
        self.assertIn(("model2", prompt_hash2), adapter._model_cache)
        self.assertIn(("model3", prompt_hash3), adapter._model_cache)
        self.assertIn(("model4", prompt_hash4), adapter._model_cache)

    @patch("google.genai.Client")
    def test_prompt_hashing(self, mock_client_class):
        """Different prompts should produce different hashes"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key="test-key")

        hash1 = adapter._hash_prompt("prompt1")
        hash2 = adapter._hash_prompt("prompt2")
        hash3 = adapter._hash_prompt("prompt1")  # Same as first

        # Different prompts → different hashes
        self.assertNotEqual(hash1, hash2)
        # Same prompt → same hash
        self.assertEqual(hash1, hash3)

    @patch("google.genai.Client")
    def test_thread_safety_concurrent_cache_access(self, mock_client_class):
        """Cache should be thread-safe under concurrent access"""
        import threading
        import time

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = NewGeminiAdapter(api_key="test-key")
        errors = []

        def access_cache(thread_id):
            try:
                for i in range(10):
                    model_name = f"model{i % 3}"  # Reuse some model names
                    prompt = f"prompt{thread_id}_{i}"
                    adapter.get_model(model_name, prompt)
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)

        # Create 5 threads accessing cache concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=access_cache, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # No errors should have occurred
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


if __name__ == "__main__":
    unittest.main()
