"""Gemini SDK Adapter - Abstraction layer for SDK migration

This module provides an abstraction layer between GeminiProvider and the underlying
Google Gemini SDK. It allows us to migrate from google.generativeai to google.genai
in a phased approach without breaking existing functionality.

Phase 1 (Current): LegacyGeminiAdapter wraps google.generativeai
Phase 2: NewGeminiAdapter implements google.genai
Phase 3: Switch default to NewGeminiAdapter
Phase 4: Remove LegacyGeminiAdapter

Issue: #133 (Gemini SDK Migration)
"""

import hashlib
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Generator, Optional


class GeminiSDKAdapter(ABC):
    """Abstract base class for Gemini SDK adapters

    This defines the interface that both legacy and new SDK adapters must implement.
    """

    @abstractmethod
    def get_model(self, model_name: str, system_instruction: Optional[str] = None) -> Any:
        """Get or create a model instance

        Args:
            model_name: Name of the Gemini model
            system_instruction: Optional system instruction/prompt

        Returns:
            Model instance (SDK-specific type)
        """
        pass

    @abstractmethod
    @contextmanager
    def handle_api_errors(self) -> Generator[None, None, None]:
        """Context manager to handle SDK-specific exceptions

        This allows the adapter to catch SDK-specific exceptions and convert them
        to standard exceptions (e.g., ValueError) that the provider layer understands.

        Example:
            with adapter.handle_api_errors():
                response = model.generate_content(...)
        """
        pass


class LegacyGeminiAdapter(GeminiSDKAdapter):
    """Adapter for google.generativeai (deprecated SDK)

    This adapter wraps the deprecated google.generativeai SDK to maintain
    backward compatibility during the migration to google.genai.

    Features:
        - Thread-safe LRU caching for models with system instructions
        - Separate default model cache for no-system-instruction case
        - Hash-based cache key with collision detection
    """

    def __init__(self, api_key: str):
        """Initialize the legacy adapter

        Args:
            api_key: Google API key for authentication
        """
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.genai = genai
        # Cache for models without system instruction: {model_name: model}
        self._default_models: dict[str, Any] = {}
        # LRU cache for models with system instruction: {(model_name, prompt_hash): (prompt, model)}
        self._models_cache: OrderedDict[tuple[str, str], tuple[str, Any]] = OrderedDict()
        self._cache_max_size = 10
        self._cache_lock = threading.Lock()

    def get_model(self, model_name: str, system_instruction: Optional[str] = None) -> Any:
        """Get or create a cached Gemini model instance

        Args:
            model_name: Name of the Gemini model
            system_instruction: Optional system instruction for the model

        Returns:
            GenerativeModel instance from google.generativeai
        """
        # If no system instruction, cache by model_name only
        if not system_instruction or not system_instruction.strip():
            with self._cache_lock:
                if model_name not in self._default_models:
                    self._default_models[model_name] = self.genai.GenerativeModel(model_name)
                return self._default_models[model_name]

        # For system instructions, use LRU cache with (model_name, prompt_hash) key
        prompt_hash = self._hash_prompt(system_instruction)
        cache_key = (model_name, prompt_hash)

        with self._cache_lock:
            if cache_key in self._models_cache:
                # Verify prompt hasn't changed (hash collision check)
                cached_prompt, cached_model = self._models_cache[cache_key]
                if cached_prompt == system_instruction:
                    # Move to end (most recently used)
                    self._models_cache.move_to_end(cache_key)
                    return cached_model

            # Create new model with system instruction
            model = self.genai.GenerativeModel(model_name, system_instruction=system_instruction)
            self._models_cache[cache_key] = (system_instruction, model)

            # LRU eviction if cache is full
            if len(self._models_cache) > self._cache_max_size:
                self._models_cache.popitem(last=False)

            return model

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """Generate SHA256 hash for a prompt to use as cache key

        Args:
            prompt: System instruction text

        Returns:
            SHA256 hash as hexadecimal string
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @contextmanager
    def handle_api_errors(self) -> Generator[None, None, None]:
        """Handle google.generativeai specific exceptions

        Catches BlockedPromptException and converts to ValueError.
        """
        try:
            yield
        except Exception as e:
            # Check if it's a BlockedPromptException by class name
            # (avoid direct type check as it may not be defined in test environment)
            if type(e).__name__ == "BlockedPromptException":
                raise ValueError(f"Prompt was blocked due to safety concerns: {e}") from e
            # Re-raise other exceptions unchanged
            raise
