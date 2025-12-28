"""Tests for concurrent safety of LLM providers

These tests verify that:
1. Multiple threads can safely call Gemini with different system prompts
2. ChatGPT client can handle concurrent requests
3. Model cache operations are thread-safe
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

from multi_llm_chat.llm_provider import ChatGPTProvider, GeminiProvider


class TestGeminiConcurrentSafety(unittest.TestCase):
    """Test concurrent safety for GeminiProvider"""

    @patch("multi_llm_chat.llm_provider.genai")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    def test_gemini_concurrent_different_prompts(self, mock_genai):
        """Test that concurrent requests with different system prompts don't interfere"""
        import time

        # Setup mock
        mock_model_class = MagicMock()
        mock_genai.GenerativeModel = mock_model_class

        # Track which prompts were used to create models (with thread safety check)
        created_models = []
        created_prompts = []

        def track_model_creation(model_name, system_instruction=None):
            # Simulate race condition by adding delay
            time.sleep(0.01)
            mock_model = MagicMock()
            # Return the system_instruction in the response to verify correct prompt was used
            mock_chunk = MagicMock()
            mock_chunk.text = f"Response with prompt: {system_instruction}"
            mock_model.generate_content = MagicMock(return_value=iter([mock_chunk]))

            # Track creation without lock to detect race conditions
            created_models.append(mock_model)
            created_prompts.append(system_instruction)
            return mock_model

        mock_model_class.side_effect = track_model_creation

        provider = GeminiProvider()

        # Define different system prompts
        prompts = [
            "You are a helpful assistant",
            "You are a coding expert",
            "You are a translator",
            "You are a writer",
            "You are a scientist",
        ]

        # Results storage
        results = {}
        errors = []
        results_lock = threading.Lock()

        def call_with_prompt(prompt_text, thread_id):
            """Call API with specific prompt and store result"""
            try:
                history = [{"role": "user", "content": f"Hello from thread {thread_id}"}]
                # Call API and consume generator
                chunks = list(provider.call_api(history, system_prompt=prompt_text))
                response_text = "".join(provider.extract_text_from_chunk(c) for c in chunks)

                with results_lock:
                    results[thread_id] = {
                        "prompt": prompt_text,
                        "response": response_text,
                        "chunks": len(chunks),
                    }
            except Exception as e:
                with results_lock:
                    errors.append({"thread_id": thread_id, "error": str(e)})

        # Launch concurrent threads
        threads = []
        for i, prompt in enumerate(prompts):
            thread = threading.Thread(target=call_with_prompt, args=(prompt, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors occurred during concurrent execution: {errors}")

        # Verify all threads completed successfully
        self.assertEqual(len(results), len(prompts), "Not all threads completed successfully")

        # CRITICAL: Verify each thread got response matching its prompt
        # This detects prompt pollution where wrong system prompt is used
        for thread_id, result in results.items():
            expected_prompt = result["prompt"]
            response = result["response"]
            self.assertIn(
                expected_prompt,
                response,
                f"Thread {thread_id} did not get response with correct prompt. "
                f"Expected '{expected_prompt}' in response, got: {response}",
            )

    @patch("multi_llm_chat.llm_provider.genai")
    @patch("multi_llm_chat.llm_provider.GOOGLE_API_KEY", "test-key")
    def test_gemini_cache_thread_safety(self, mock_genai):
        """Test that cache operations are thread-safe during concurrent access"""
        import time

        mock_model_class = MagicMock()
        mock_genai.GenerativeModel = mock_model_class

        # Track cache state changes to detect race conditions
        cache_operations = []
        operation_lock = threading.Lock()

        # Create a mock model that simulates some processing time
        def create_slow_model(model_name, system_instruction=None):
            mock_model = MagicMock()
            mock_model.generate_content = MagicMock(return_value=iter([MagicMock(text="response")]))
            # Simulate model creation taking time to increase chance of race
            time.sleep(0.02)

            # Track that a model was created (should detect duplicate creations)
            with operation_lock:
                cache_operations.append(("create", system_instruction))

            return mock_model

        mock_model_class.side_effect = create_slow_model

        provider = GeminiProvider()

        # Use the same prompt from multiple threads to stress test cache
        shared_prompt = "You are a helpful assistant"
        call_count = 20
        errors = []
        completed = []
        completed_lock = threading.Lock()

        def make_call(call_id):
            try:
                history = [{"role": "user", "content": f"Call {call_id}"}]
                list(provider.call_api(history, system_prompt=shared_prompt))
                with completed_lock:
                    completed.append(call_id)
            except Exception as e:
                with completed_lock:
                    errors.append({"call_id": call_id, "error": str(e)})

        # Launch many concurrent threads using the same prompt
        threads = []
        for i in range(call_count):
            thread = threading.Thread(target=make_call, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors and all calls completed
        self.assertEqual(len(errors), 0, f"Errors during cache stress test: {errors}")
        self.assertEqual(len(completed), call_count, "Not all calls completed")

        # CRITICAL: With proper caching, the same prompt should only create model once
        # Count how many times the same prompt triggered model creation
        creates_for_shared_prompt = sum(
            1 for op, prompt in cache_operations if op == "create" and prompt == shared_prompt
        )

        # Without thread safety, we might see multiple creations for the same prompt
        # With proper locking, we should see only 1 creation
        self.assertEqual(
            creates_for_shared_prompt,
            1,
            f"Expected 1 model creation for shared prompt, got {creates_for_shared_prompt}. "
            "This indicates a race condition in cache access.",
        )


class TestChatGPTConcurrentSafety(unittest.TestCase):
    """Test concurrent safety for ChatGPTProvider"""

    @patch("multi_llm_chat.llm_provider.openai.OpenAI")
    @patch("multi_llm_chat.llm_provider.OPENAI_API_KEY", "test-key")
    def test_chatgpt_concurrent_requests(self, mock_openai_class):
        """Test that ChatGPT client handles concurrent requests safely"""
        # Setup mock client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response
        def create_mock_stream(*args, **kwargs):
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = "response"
            return iter([mock_chunk])

        mock_client.chat.completions.create = MagicMock(side_effect=create_mock_stream)

        provider = ChatGPTProvider()

        # Results storage
        results = []
        errors = []

        def make_request(request_id):
            try:
                history = [{"role": "user", "content": f"Request {request_id}"}]
                system_prompt = f"System prompt {request_id}"
                chunks = list(provider.call_api(history, system_prompt=system_prompt))
                results.append({"id": request_id, "chunks": len(chunks)})
            except Exception as e:
                errors.append({"id": request_id, "error": str(e)})

        # Launch concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors
        self.assertEqual(len(errors), 0, f"Errors during concurrent requests: {errors}")
        self.assertEqual(len(results), 10, "Not all requests completed")


if __name__ == "__main__":
    unittest.main()
