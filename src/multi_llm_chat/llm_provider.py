"""LLM Provider abstraction layer using Strategy pattern

This module provides a unified interface for different LLM providers (Gemini, ChatGPT, etc.),
making it easy to add new providers without modifying existing code.
"""

import hashlib
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict

import google.generativeai as genai
import openai
from dotenv import load_dotenv

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from google.generativeai.types import BlockedPromptException
from .token_utils import get_buffer_factor, estimate_tokens, get_max_context_length

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo")


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def call_api(self, history, system_prompt=None):
        """Call the LLM API and return a generator of response chunks

        Args:
            history: List of conversation history dicts with 'role' and 'content'
            system_prompt: Optional system instruction

        Yields:
            Response chunks from the API
        """
        pass

    @abstractmethod
    def extract_text_from_chunk(self, chunk):
        """Extract text content from a response chunk

        Args:
            chunk: Response chunk from the API

        Returns:
            str: Extracted text content
        """
        pass

    @abstractmethod
    def get_token_info(self, text, history=None, model_name=None):
        """Get token usage information

        Args:
            text: Text to analyze
            history: Optional conversation history
            model_name: Optional specific model name (uses default if None)

        Returns:
            dict: Token information with keys 'input_tokens', 'max_tokens'
        """
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider with LRU caching for models"""

    def __init__(self):
        self._default_model = None
        self._models_cache = OrderedDict()  # LRU cache: hash -> (prompt, model)
        self._cache_max_size = 10  # Limit cache size to prevent memory leak
        self._configure()

    def _configure(self):
        """Configure the Gemini SDK if an API key is available"""
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            return True
        return False

    @staticmethod
    def _hash_prompt(prompt):
        """Generate SHA256 hash for a prompt to use as cache key"""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _get_model(self, system_prompt=None):
        """Get or create a cached Gemini model instance with LRU eviction

        Args:
            system_prompt: Optional system instruction for the model

        Returns:
            GenerativeModel instance
        """
        # If no system prompt, use the default cached model
        if not system_prompt or not system_prompt.strip():
            if self._default_model is None:
                self._default_model = genai.GenerativeModel(GEMINI_MODEL)
            return self._default_model

        # For system prompts, use LRU cache with hash key
        prompt_hash = self._hash_prompt(system_prompt)

        if prompt_hash in self._models_cache:
            # Verify prompt hasn't changed (hash collision check)
            cached_prompt, cached_model = self._models_cache[prompt_hash]
            if cached_prompt == system_prompt:
                # Move to end (most recently used)
                self._models_cache.move_to_end(prompt_hash)
                return cached_model
            # Hash collision detected - explicitly evict old entry
            del self._models_cache[prompt_hash]

        # Create new model and add to cache
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
        self._models_cache[prompt_hash] = (system_prompt, model)

        # Evict oldest if cache is full
        if len(self._models_cache) > self._cache_max_size:
            self._models_cache.popitem(last=False)

        return model

    @staticmethod
    def format_history(history):
        """Convert history to Gemini API format

        Filters out responses from other LLMs (e.g., ChatGPT) to avoid
        sending Gemini messages it didn't generate, which would create
        a self-contradictory conversation.
        """
        gemini_history = []
        for entry in history:
            role = entry["role"]
            # Only include user messages and Gemini's own responses
            if role == "user":
                gemini_history.append({"role": "user", "parts": [entry["content"]]})
            elif role == "gemini":
                gemini_history.append({"role": "model", "parts": [entry["content"]]})
            # Skip chatgpt, system, and other roles - they shouldn't be sent to Gemini
        return gemini_history

    def call_api(self, history, system_prompt=None):
        """Call Gemini API and yield response chunks with safety error handling"""
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set")

        # Get cached or new model with system prompt
        model = self._get_model(system_prompt)

        # Filter history to only include user and Gemini messages
        gemini_history = self.format_history(history)

        # Call API with streaming, handling BlockedPromptException
        try:
            response = model.generate_content(gemini_history, stream=True)
            yield from response
        except genai.types.BlockedPromptException as e:
            raise ValueError(f"Prompt was blocked due to safety concerns: {e}") from e

    def extract_text_from_chunk(self, chunk):
        """Extract text from Gemini response chunk"""
        return getattr(chunk, "text", "")

    def get_token_info(self, text, history=None, model_name=None):
        """Get token information for Gemini

        Uses estimation with buffer factor (from TOKEN_ESTIMATION_BUFFER_FACTOR env var).
        """
        # Use provided model name or fall back to default
        effective_model = model_name if model_name else GEMINI_MODEL

        # Get buffer factor from environment variable
        buffer_factor = get_buffer_factor()

        # Calculate tokens for system prompt/text
        token_count = int(estimate_tokens(text) * buffer_factor)

        # Add history tokens if provided (only count user and gemini messages)
        if history:
            for entry in history:
                role = entry.get("role", "")
                if role in {"user", "gemini"}:
                    content = entry.get("content", "")
                    token_count += int(estimate_tokens(content) * buffer_factor)

        # Get max context length for this model
        max_context = get_max_context_length(effective_model)

        return {
            "input_tokens": token_count,
            "max_tokens": max_context,
        }


class ChatGPTProvider(LLMProvider):
    """OpenAI ChatGPT LLM provider"""

    def __init__(self):
        self._client = None
        if OPENAI_API_KEY:
            self._client = openai.OpenAI(api_key=OPENAI_API_KEY)

    @staticmethod
    def format_history(history):
        """Convert history to ChatGPT API format

        Filters out responses from other LLMs (e.g., Gemini) to avoid
        sending ChatGPT messages it didn't generate, which would create
        a self-contradictory conversation.
        """
        chatgpt_history = []
        for entry in history:
            role = entry["role"]
            if role == "system":
                chatgpt_history.append({"role": "system", "content": entry["content"]})
            elif role == "user":
                chatgpt_history.append({"role": "user", "content": entry["content"]})
            elif role == "chatgpt":
                chatgpt_history.append({"role": "assistant", "content": entry["content"]})
            # Skip gemini and other roles - they shouldn't be sent to ChatGPT
        return chatgpt_history

    def call_api(self, history, system_prompt=None):
        """Call ChatGPT API and yield response chunks"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")

        # Build messages for OpenAI format
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        # Filter history to only include user and ChatGPT messages
        chatgpt_history = self.format_history(history)
        messages.extend(chatgpt_history)

        # Call API with streaming
        stream = self._client.chat.completions.create(
            model=CHATGPT_MODEL, messages=messages, stream=True
        )
        yield from stream

    def extract_text_from_chunk(self, chunk):
        """Extract text from ChatGPT response chunk

        Handles both string and list responses from OpenAI API.
        """
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            delta_content = getattr(delta, "content", None)

            # Handle both string and list responses from OpenAI API
            if isinstance(delta_content, list):
                return "".join(
                    part.text if hasattr(part, "text") else str(part) for part in delta_content
                )
            elif delta_content is not None:
                return delta_content
        return ""

    def get_token_info(self, text, history=None, model_name=None):
        """Get token information for ChatGPT

        Uses tiktoken for accurate counting if available, falls back to estimation.
        """
        # Use provided model name or fall back to default
        effective_model = model_name if model_name else CHATGPT_MODEL
        token_count = 0
        use_estimation = not TIKTOKEN_AVAILABLE

        # Try tiktoken for accurate counting if available
        if TIKTOKEN_AVAILABLE:
            try:
                # Map model name to tiktoken encoding
                model_lower = effective_model.lower()
                if "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower:
                    encoding = tiktoken.get_encoding("o200k_base")
                elif "gpt-4" in model_lower:
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:  # gpt-3.5 and others
                    encoding = tiktoken.get_encoding("cl100k_base")

                # Count system prompt/text tokens
                token_count = len(encoding.encode(text))

                # Add message overhead (3 tokens per message for OpenAI spec)
                token_count += 3

                # Add history tokens if provided (only count user and chatgpt messages)
                if history:
                    for entry in history:
                        role = entry.get("role", "")
                        if role in {"user", "chatgpt"}:
                            content = entry.get("content", "")
                            token_count += len(encoding.encode(content)) + 3

            except Exception:
                # Fall back to estimation if tiktoken fails
                use_estimation = True

        # Use estimation with buffer (if tiktoken unavailable or failed)
        if use_estimation:
            buffer_factor = get_buffer_factor()
            token_count = int(estimate_tokens(text) * buffer_factor)
            if history:
                for entry in history:
                    if entry.get("role") in {"user", "chatgpt"}:
                        token_count += int(
                            estimate_tokens(entry.get("content", "")) * buffer_factor
                        )

        # Get max context length for this model
        max_context = get_max_context_length(effective_model)

        return {
            "input_tokens": token_count,
            "max_tokens": max_context,
        }


# Provider registry
_PROVIDERS = {"gemini": GeminiProvider, "chatgpt": ChatGPTProvider}

# Cache provider instances for reuse
_PROVIDER_INSTANCES = {}
_provider_lock = threading.Lock()


def get_provider(provider_name):
    """Factory function to get a provider instance (thread-safe)

    Returns cached instance if available to reuse API clients and models.
    Thread-safe for concurrent access in WebUI environment.

    Args:
        provider_name: Name of the provider ('gemini', 'chatgpt', etc.)

    Returns:
        LLMProvider: Instance of the requested provider

    Raises:
        ValueError: If provider_name is not registered
    """
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    # Thread-safe check-and-create pattern
    with _provider_lock:
        if provider_name not in _PROVIDER_INSTANCES:
            provider_class = _PROVIDERS[provider_name]
            _PROVIDER_INSTANCES[provider_name] = provider_class()

    return _PROVIDER_INSTANCES[provider_name]
