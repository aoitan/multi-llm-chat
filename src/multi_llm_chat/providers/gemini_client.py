"""NewGeminiAdapter - google.genai SDK implementation

This adapter implements the GeminiSDKAdapter interface using the new google.genai SDK.
Issue: #137 (Phase 2: New SDK Implementation)
"""

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
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with google.genai.Client

        Args:
            api_key: Google API key (can be None for testing)
        """
        self.client = genai.Client(api_key=api_key)
        self._model_cache: dict[tuple[str, Optional[str]], _ModelProxy] = {}

    def get_model(self, model_name: str, system_instruction: Optional[str] = None) -> _ModelProxy:
        """Get model proxy that wraps new SDK client

        Args:
            model_name: Name of the Gemini model
            system_instruction: Optional system instruction

        Returns:
            _ModelProxy that looks like old SDK model
        """
        cache_key = (model_name, system_instruction)
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = _ModelProxy(self.client, model_name, system_instruction)
        return self._model_cache[cache_key]

    @contextmanager
    def handle_api_errors(self) -> Generator[None, None, None]:
        """Context manager for SDK-specific error handling

        The new SDK may have different exception types than the legacy SDK.
        For now, we pass through all exceptions as-is.
        """
        yield
