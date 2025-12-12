"""LLM Provider abstraction layer using Strategy pattern

This module provides a unified interface for different LLM providers (Gemini, ChatGPT, etc.),
making it easy to add new providers without modifying existing code.
"""

import hashlib
import logging
import os
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

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo")


def _estimate_tokens(text):
    """Estimate token count for mixed English/Japanese text

    More accurate estimation that accounts for Japanese characters:
    - ASCII/Latin: ~4 chars = 1 token
    - Japanese (hiragana/katakana/kanji): ~1.5 chars = 1 token

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Count Japanese characters (Unicode ranges for CJK)
    japanese_chars = sum(
        1
        for char in text
        if "\u3040" <= char <= "\u309f"  # Hiragana
        or "\u30a0" <= char <= "\u30ff"  # Katakana
        or "\u4e00" <= char <= "\u9fff"  # Kanji
        or "\uff00" <= char <= "\uffef"  # Full-width characters
    )

    ascii_chars = len(text) - japanese_chars

    # Japanese: 1.5 chars ≈ 1 token, ASCII: 4 chars ≈ 1 token
    estimated = (japanese_chars / 1.5) + (ascii_chars / 4.0)

    return int(estimated)


def _get_max_context_length(model_name):
    """Get maximum context length for the specified model

    Reads from environment variables with fallback to model defaults

    Args:
        model_name: Model identifier

    Returns:
        Maximum context length in tokens
    """
    model_lower = model_name.lower()

    # Check for model-specific environment variable
    if "gemini" in model_lower:
        gemini_max = os.getenv("GEMINI_MAX_CONTEXT_LENGTH")
        if gemini_max:
            try:
                return int(gemini_max)
            except ValueError:
                logging.warning(f"Invalid GEMINI_MAX_CONTEXT_LENGTH: {gemini_max}. Using default.")

    if "gpt" in model_lower:
        chatgpt_max = os.getenv("CHATGPT_MAX_CONTEXT_LENGTH")
        if chatgpt_max:
            try:
                return int(chatgpt_max)
            except ValueError:
                logging.warning(
                    f"Invalid CHATGPT_MAX_CONTEXT_LENGTH: {chatgpt_max}. Using default."
                )

    # Fall back to default
    default_max = os.getenv("DEFAULT_MAX_CONTEXT_LENGTH")
    if default_max:
        try:
            return int(default_max)
        except ValueError:
            logging.warning(f"Invalid DEFAULT_MAX_CONTEXT_LENGTH: {default_max}. Using default.")

    # Built-in model-specific defaults
    MODEL_DEFAULTS = [
        ("gemini-2.0-flash", 1048576),
        ("gemini-exp-1206", 1048576),
        ("gemini-1.5-pro", 2097152),
        ("gemini-1.5-flash", 1048576),
        ("gemini-pro", 32760),
        ("gemini", 32760),
        ("gpt-4o", 128000),
        ("gpt-4-turbo", 128000),
        ("gpt-4-1106", 128000),
        ("gpt-4", 8192),
        ("gpt-3.5-turbo-16k", 16385),
        ("gpt-3.5", 4096),
    ]

    # Find first matching pattern
    for pattern, context_length in MODEL_DEFAULTS:
        if pattern in model_lower:
            return context_length

    return 4096


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
    def get_token_info(self, text, history=None):
        """Get token usage information

        Args:
            text: Text to analyze
            history: Optional conversation history

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
            # Hash collision - evict and recreate

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

    def get_token_info(self, text, history=None):
        """Get token information for Gemini

        Uses estimation with buffer factor for Gemini models.
        """
        # Calculate tokens for system prompt/text
        token_count = int(_estimate_tokens(text) * 1.2)  # 20% buffer

        # Add history tokens if provided (only count user and gemini messages)
        if history:
            for entry in history:
                role = entry.get("role", "")
                if role in {"user", "gemini"}:
                    content = entry.get("content", "")
                    token_count += int(_estimate_tokens(content) * 1.2)

        # Get max context length for this model
        max_context = _get_max_context_length(GEMINI_MODEL)

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

    def get_token_info(self, text, history=None):
        """Get token information for ChatGPT

        Uses tiktoken for accurate counting if available, falls back to estimation.
        """
        token_count = 0

        # Use tiktoken for accurate counting if available
        if TIKTOKEN_AVAILABLE:
            try:
                # Map model name to tiktoken encoding
                model_lower = CHATGPT_MODEL.lower()
                if "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower:
                    encoding = tiktoken.get_encoding("o200k_base")
                elif "gpt-4" in model_lower:
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:  # gpt-3.5 and others
                    encoding = tiktoken.get_encoding("cl100k_base")

                # Count system prompt/text tokens
                token_count = len(encoding.encode(text))

                # Add message overhead (4 tokens per message)
                token_count += 4

                # Add history tokens if provided (only count user and chatgpt messages)
                if history:
                    for entry in history:
                        role = entry.get("role", "")
                        if role in {"user", "chatgpt"}:
                            content = entry.get("content", "")
                            token_count += len(encoding.encode(content)) + 4

            except Exception:
                # Fall back to estimation if tiktoken fails
                token_count = int(_estimate_tokens(text) * 1.2)
                if history:
                    for entry in history:
                        if entry.get("role") in {"user", "chatgpt"}:
                            token_count += int(_estimate_tokens(entry.get("content", "")) * 1.2)
        else:
            # Use estimation with buffer
            token_count = int(_estimate_tokens(text) * 1.2)
            if history:
                for entry in history:
                    if entry.get("role") in {"user", "chatgpt"}:
                        token_count += int(_estimate_tokens(entry.get("content", "")) * 1.2)

        # Get max context length for this model
        max_context = _get_max_context_length(CHATGPT_MODEL)

        return {
            "input_tokens": token_count,
            "max_tokens": max_context,
        }


# Provider registry
_PROVIDERS = {"gemini": GeminiProvider, "chatgpt": ChatGPTProvider}

# Cache provider instances for reuse
_PROVIDER_INSTANCES = {}


def get_provider(provider_name):
    """Factory function to get a provider instance

    Returns cached instance if available to reuse API clients and models.

    Args:
        provider_name: Name of the provider ('gemini', 'chatgpt', etc.)

    Returns:
        LLMProvider: Instance of the requested provider

    Raises:
        ValueError: If provider_name is not registered
    """
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    # Return cached instance if exists
    if provider_name not in _PROVIDER_INSTANCES:
        provider_class = _PROVIDERS[provider_name]
        _PROVIDER_INSTANCES[provider_name] = provider_class()

    return _PROVIDER_INSTANCES[provider_name]
