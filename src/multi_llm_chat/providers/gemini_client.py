"""NewGeminiAdapter - google.genai SDK implementation

This adapter implements the GeminiSDKAdapter interface using the new google.genai SDK.
Issue: #137 (Phase 2: New SDK Implementation)
"""

import hashlib
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import google.genai as genai

from .gemini_adapter import GeminiSDKAdapter


class _ModelProxy:
    """Proxy object that wraps new SDK client to look like old SDK model

    This allows GeminiProvider.call_api() to work with both SDKs
    without modification.
    """

    def __init__(self, client: genai.Client, model_name: str, system_instruction: Optional[str]):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction

    def generate_content(
        self, contents: List[Dict[str, Any]], stream: bool = False, tools: Optional[List] = None
    ):
        """Generate content using new SDK client API

        Args:
            contents: Conversation history in Gemini format
            stream: Whether to stream the response
            tools: Optional list of tools

        Returns:
            Response or streaming response from new SDK
        """
        # Build config with system instruction if provided
        config = {}
        if self._system_instruction:
            config["system_instruction"] = self._system_instruction
        if tools:
            config["tools"] = tools

        # Only pass config if it has values
        config_arg = config if config else None

        if stream:
            return self._client.models.generate_content_stream(
                model=self._model_name, contents=contents, config=config_arg
            )
        else:
            return self._client.models.generate_content(
                model=self._model_name, contents=contents, config=config_arg
            )


class NewGeminiAdapter(GeminiSDKAdapter):
    """Adapter for google.genai (new SDK)

    The new SDK uses a Client-based API instead of module-level functions.

    Features:
        - Thread-safe LRU caching for model proxies
        - Hash-based cache key with system instruction hashing
        - Cache size limit to prevent unbounded growth
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with google.genai.Client

        Args:
            api_key: Google API key (can be None for testing)
        """
        self.client = genai.Client(api_key=api_key)
        # LRU cache: {(model_name, prompt_hash): proxy}
        self._model_cache: OrderedDict[tuple[str, str], _ModelProxy] = OrderedDict()
        self._cache_max_size = 10
        self._cache_lock = threading.Lock()

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """Hash system instruction for cache key

        Args:
            prompt: System instruction text

        Returns:
            SHA256 hash of the prompt (full 64 chars for collision safety)
        """
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get_model(self, model_name: str, system_instruction: Optional[str] = None) -> _ModelProxy:
        """Get model proxy that wraps new SDK client

        Args:
            model_name: Name of the Gemini model
            system_instruction: Optional system instruction

        Returns:
            _ModelProxy that looks like old SDK model
        """
        # Hash system instruction for cache key
        prompt_hash = self._hash_prompt(system_instruction) if system_instruction else ""
        cache_key = (model_name, prompt_hash)

        with self._cache_lock:
            # Check if cached
            if cache_key in self._model_cache:
                # Move to end (LRU update)
                self._model_cache.move_to_end(cache_key)
                return self._model_cache[cache_key]

            # Create new proxy
            proxy = _ModelProxy(self.client, model_name, system_instruction)
            self._model_cache[cache_key] = proxy

            # Enforce LRU limit
            if len(self._model_cache) > self._cache_max_size:
                self._model_cache.popitem(last=False)  # Remove oldest

            return proxy

    @contextmanager
    def handle_api_errors(self) -> Generator[None, None, None]:
        """Context manager for SDK-specific error handling

        Handles new SDK exceptions and converts safety blocks to ValueError
        for consistency with LegacyGeminiAdapter.
        """
        try:
            yield
        except Exception as e:
            # Check for ClientError with safety blocking (by status code or message)
            if type(e).__name__ in ("ClientError", "APIError"):
                error_msg = str(e).lower()
                # New SDK reports safety blocks in error messages
                if any(keyword in error_msg for keyword in ["safety", "blocked", "harm"]):
                    raise ValueError(f"Prompt was blocked due to safety concerns: {e}") from e
            # Re-raise other exceptions unchanged
            raise
