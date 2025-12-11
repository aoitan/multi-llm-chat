"""LLM Provider abstraction layer using Strategy pattern

This module provides a unified interface for different LLM providers (Gemini, ChatGPT, etc.),
making it easy to add new providers without modifying existing code.
"""

import os
from abc import ABC, abstractmethod

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
    """Google Gemini LLM provider"""

    def __init__(self):
        self._model = None
        self._configure()

    def _configure(self):
        """Configure the Gemini SDK if an API key is available"""
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            return True
        return False

    def call_api(self, history, system_prompt=None):
        """Call Gemini API and yield response chunks"""
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set")

        # Import here to avoid circular dependency
        from multi_llm_chat.core import format_history_for_gemini

        # Create model with system instruction if provided
        if system_prompt and system_prompt.strip():
            model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
        else:
            if self._model is None:
                self._model = genai.GenerativeModel(GEMINI_MODEL)
            model = self._model

        # Filter history to only include user and Gemini messages
        gemini_history = format_history_for_gemini(history)

        # Call API with streaming
        response = model.generate_content(gemini_history, stream=True)
        yield from response

    def extract_text_from_chunk(self, chunk):
        """Extract text from Gemini response chunk"""
        return getattr(chunk, "text", "")

    def get_token_info(self, text, history=None):
        """Get token information for Gemini"""
        # Simplified token counting - full implementation would use Gemini's API
        # For MVP, return estimated values
        estimated_tokens = len(text.split())
        return {"input_tokens": estimated_tokens, "max_tokens": 1048576}


class ChatGPTProvider(LLMProvider):
    """OpenAI ChatGPT LLM provider"""

    def __init__(self):
        self._client = None
        if OPENAI_API_KEY:
            self._client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def call_api(self, history, system_prompt=None):
        """Call ChatGPT API and yield response chunks"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")

        # Import here to avoid circular dependency
        from multi_llm_chat.core import format_history_for_chatgpt

        # Build messages for OpenAI format
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        # Filter history to only include user and ChatGPT messages
        chatgpt_history = format_history_for_chatgpt(history)
        messages.extend(chatgpt_history)

        # Call API with streaming
        stream = self._client.chat.completions.create(
            model=CHATGPT_MODEL, messages=messages, stream=True
        )
        yield from stream

    def extract_text_from_chunk(self, chunk):
        """Extract text from ChatGPT response chunk"""
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            return getattr(delta, "content", "") or ""
        return ""

    def get_token_info(self, text, history=None):
        """Get token information for ChatGPT"""
        # Simplified token counting
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(CHATGPT_MODEL)
                tokens = len(encoding.encode(text))
                return {"input_tokens": tokens, "max_tokens": 128000}
            except Exception:
                pass

        # Fallback estimation
        estimated_tokens = len(text.split())
        return {"input_tokens": estimated_tokens, "max_tokens": 128000}


# Provider registry
_PROVIDERS = {"gemini": GeminiProvider, "chatgpt": ChatGPTProvider}


def get_provider(provider_name):
    """Factory function to get a provider instance

    Args:
        provider_name: Name of the provider ('gemini', 'chatgpt', etc.)

    Returns:
        LLMProvider: Instance of the requested provider

    Raises:
        ValueError: If provider_name is not registered
    """
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    provider_class = _PROVIDERS[provider_name]
    return provider_class()
