"""Base classes for LLM providers

This module defines the abstract interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def call_api(self, history, system_prompt=None, tools: Optional[List[Dict[str, Any]]] = None):
        """Call the LLM API and return a generator of response chunks

        Args:
            history: List of conversation history dicts with 'role' and 'content'.
                     MUST NOT be mutated by the implementation. Implementations
                     should create a copy via format_history() if modifications
                     are needed.
            system_prompt: Optional system instruction
            tools: Optional list of tools for the LLM

        Yields:
            Response chunks from the API
        """
        pass

    @abstractmethod
    def extract_text_from_chunk(self, chunk):
        """Extract text content from a response chunk

        Args:
            chunk: A response chunk from the API

        Returns:
            str: Text content from the chunk
        """
        pass

    @abstractmethod
    def get_token_info(self, text, history=None, model_name=None, has_tools=False):
        """Get token count information

        Args:
            text: Text to count tokens for
            history: Optional conversation history
            model_name: Optional model name override
            has_tools: Whether tools are being used

        Returns:
            dict: Token information with 'input_tokens' and 'max_tokens'
        """
        pass

    @staticmethod
    @abstractmethod
    def format_history(history):
        """Convert history to provider-specific API format

        Args:
            history: Universal history format

        Returns:
            Provider-specific history format
        """
        pass

    def stream_text_events(self, history, system_prompt=None):
        """Stream text events from API responses (filters empty strings)

        Args:
            history: Conversation history
            system_prompt: Optional system instruction

        Yields:
            str: Non-empty text content from response chunks
        """
        for chunk in self.call_api(history, system_prompt):
            text = self.extract_text_from_chunk(chunk)
            if text:  # Filter out empty strings
                yield text
