"""Gemini SDK Adapter - Abstraction layer for SDK

This module provides an abstraction layer between GeminiProvider and the underlying
Google Gemini SDK (google.genai).

Phase 4 complete: LegacyGeminiAdapter (google.generativeai) has been removed.
Only NewGeminiAdapter (google.genai) is used. See gemini_client.py.

Issue: #133 (Gemini SDK Migration), #139 (Phase 4: Remove Legacy SDK)
"""

from abc import ABC, abstractmethod
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
