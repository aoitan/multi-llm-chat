"""Tests for concurrent safety of LLM providers

These tests verify that:
1. Multiple threads can safely call Gemini with different system prompts
2. ChatGPT client can handle concurrent requests
3. Model cache operations are thread-safe
4. Session-scoped providers are isolated from each other
"""

import asyncio
import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest

from multi_llm_chat.llm_provider import ChatGPTProvider


async def consume_async_gen(gen):
    """Helper to consume a generator (sync or async) and return all yielded items."""
    results = []
    # Check if it's an async generator
    if hasattr(gen, "__aiter__"):
        async for item in gen:
            results.append(item)
    else:
        # Synchronous generator
        for item in gen:
            results.append(item)
    return results


class TestChatGPTConcurrentSafety(unittest.TestCase):
    """Test concurrent safety for ChatGPTProvider"""

    @pytest.mark.skip(reason="Needs fixing for sync generator compatibility after merge with main")
    @patch("multi_llm_chat.providers.openai.openai.OpenAI")
    def test_chatgpt_concurrent_requests(self, mock_openai_class):
        """Test that ChatGPT client handles concurrent requests safely"""
        # Setup mock client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response (synchronous generator)
        def create_mock_stream(*args, **kwargs):
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = "response"

            def mock_iter():
                yield mock_chunk

            mock_stream = MagicMock()
            mock_stream.__iter__ = mock_iter
            return mock_stream

        mock_client.chat.completions.create = create_mock_stream

        provider = ChatGPTProvider()

        # Results storage
        results = []
        errors = []

        def make_request(request_id):
            try:
                history = [{"role": "user", "content": f"Request {request_id}"}]
                system_prompt = f"System prompt {request_id}"

                async def run_test():
                    chunks = []
                    gen = provider.call_api(history, system_prompt=system_prompt)
                    # Handle both sync and async generators
                    if hasattr(gen, "__aiter__"):
                        async for c in gen:
                            chunks.append(c)
                    else:
                        for c in gen:
                            chunks.append(c)
                    return chunks

                chunks = asyncio.run(run_test())
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


class TestGeminiConcurrentSafety(unittest.TestCase):
    """Test concurrent safety for GeminiProvider with new SDK (Issue #139 migration)"""

    @patch("google.genai.Client")
    def test_gemini_concurrent_different_prompts(self, mock_client_class):
        """Concurrent requests with different system prompts should not interfere"""
        import time

        from multi_llm_chat.llm_provider import GeminiProvider

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Track which system prompts were used in generate_content_stream calls
        call_log = []
        call_log_lock = threading.Lock()

        def mock_generate_content_stream(model, contents, config=None):
            system_instruction = config.get("system_instruction") if config else None
            # Simulate some work
            time.sleep(0.01)
            text = f"Response for: {system_instruction}"
            mock_part = MagicMock(spec=["text"])
            mock_part.text = text
            mock_chunk = MagicMock()
            mock_chunk.parts = []
            mock_chunk.text = text
            with call_log_lock:
                call_log.append(system_instruction)
            return iter([mock_chunk])

        mock_client.models.generate_content_stream.side_effect = mock_generate_content_stream

        provider = GeminiProvider(api_key="test-key")

        prompts = [
            "You are a helpful assistant",
            "You are a coding expert",
            "You are a translator",
        ]

        results = {}
        errors = []
        lock = threading.Lock()

        def call_with_prompt(prompt_text, thread_id):
            try:
                history = [{"role": "user", "content": f"Hello from {thread_id}"}]

                async def run():
                    chunks = []
                    async for c in provider.call_api(history, system_prompt=prompt_text):
                        chunks.append(c)
                    return chunks

                chunks = asyncio.run(run())
                response_text = "".join(provider.extract_text_from_chunk(c) for c in chunks)
                with lock:
                    results[thread_id] = {
                        "prompt": prompt_text,
                        "response": response_text,
                        "chunks": len(chunks),
                    }
            except Exception as e:
                with lock:
                    errors.append({"thread_id": thread_id, "error": str(e)})

        threads = [
            threading.Thread(target=call_with_prompt, args=(p, i)) for i, p in enumerate(prompts)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), len(prompts))
        # 各スレッドが自分のプロンプトに対応した応答を受け取っていることを確認
        for _thread_id, result in results.items():
            expected_response = f"Response for: {result['prompt']}"
            self.assertEqual(result["response"], expected_response)

    @patch("google.genai.Client")
    def test_gemini_cache_thread_safety(self, mock_client_class):
        """Model proxy cache should be thread-safe under concurrent access"""
        import time

        from multi_llm_chat.llm_provider import GeminiProvider

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        def slow_stream(model, contents, config=None):
            time.sleep(0.005)
            mock_chunk = MagicMock()
            mock_chunk.parts = []
            mock_chunk.text = "response"
            return iter([mock_chunk])

        mock_client.models.generate_content_stream.side_effect = slow_stream

        provider = GeminiProvider(api_key="test-key")

        shared_prompt = "You are a helpful assistant"
        call_count = 10
        errors = []
        completed = []
        lock = threading.Lock()

        def make_call(call_id):
            try:
                history = [{"role": "user", "content": f"Call {call_id}"}]

                async def run():
                    async for _ in provider.call_api(history, system_prompt=shared_prompt):
                        pass

                asyncio.run(run())
                with lock:
                    completed.append(call_id)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=make_call, args=(i,)) for i in range(call_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
        self.assertEqual(len(completed), call_count)
        # 同一プロンプトに対してモデルプロキシは1個だけキャッシュされることを確認
        self.assertEqual(
            len(provider._adapter._model_cache),
            1,
            "同一プロンプトに対してモデルプロキシは一度だけキャッシュされるべきです",
        )


class TestSessionScopedProviders(unittest.TestCase):
    """Test session-scoped provider isolation with new SDK (Issue #139 migration)"""

    @patch("google.genai.Client")
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    def test_session_isolated_providers(self, mock_client_class):
        """create_provider should return different instances for different sessions"""
        from multi_llm_chat.llm_provider import create_provider

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider1 = create_provider("gemini")
        provider2 = create_provider("gemini")

        self.assertIsNot(provider1, provider2, "Sessions should have isolated provider instances")

        # Each provider has its own adapter with separate cache
        self.assertIsNotNone(
            provider1._adapter, "Adapter should be initialized when API key is set"
        )
        self.assertIsNot(
            provider1._adapter,
            provider2._adapter,
            "Each session should have its own adapter (separate model cache)",
        )

    @patch("google.genai.Client")
    def test_provider_injection_in_chatservice(self, mock_client_class):
        """ChatService should accept and use injected provider"""
        from multi_llm_chat.chat_service import ChatService
        from multi_llm_chat.llm_provider import create_provider

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = create_provider("gemini")
        service = ChatService(gemini_provider=provider)

        self.assertIs(
            service.gemini_provider,
            provider,
            "ChatService should use the injected provider",
        )


if __name__ == "__main__":
    unittest.main()
