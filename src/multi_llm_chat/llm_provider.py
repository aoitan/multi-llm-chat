"""LLM Provider abstraction layer using Strategy pattern

This module provides a unified interface for different LLM providers (Gemini, ChatGPT, etc.),
making it easy to add new providers without modifying existing code.

REFACTORING NOTE (Issue #101):
This module is being split into providers/ package. OpenAI-specific code has been moved to
providers/openai.py and Gemini-specific code to providers/gemini.py.
The classes are re-exported here for backward compatibility.
"""

import logging
import os
import threading

from dotenv import load_dotenv

# Load environment variables BEFORE importing providers to avoid race conditions
load_dotenv()

from .providers.base import LLMProvider
from .providers.gemini import (
    GeminiProvider,
    GeminiToolCallAssembler,
    _parse_tool_response_payload,
    mcp_tools_to_gemini_format,
)
from .providers.openai import (
    CHATGPT_MODEL,
    TIKTOKEN_AVAILABLE,
    ChatGPTProvider,
    OpenAIToolCallAssembler,
    mcp_tools_to_openai_format,
    parse_openai_tool_call,
)

logger = logging.getLogger(__name__)

# Environment variables (Re-exported for backward compatibility)
# Note: providers.openai does NOT export OPENAI_API_KEY to avoid early evaluation.
# We define it here after load_dotenv().
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")

# Feature flags
MCP_ENABLED = os.getenv("MULTI_LLM_CHAT_MCP_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Re-export for backward compatibility
__all__ = [
    "LLMProvider",
    "ChatGPTProvider",
    "GeminiProvider",
    "OpenAIToolCallAssembler",
    "GeminiToolCallAssembler",
    "mcp_tools_to_openai_format",
    "mcp_tools_to_gemini_format",
    "_parse_tool_response_payload",
    "parse_openai_tool_call",
    "CHATGPT_MODEL",
    "OPENAI_API_KEY",
    "TIKTOKEN_AVAILABLE",
    "GEMINI_MODEL",
    "GOOGLE_API_KEY",
    "create_provider",
    "get_provider",
]

# Provider registry
_PROVIDERS = {"gemini": GeminiProvider, "chatgpt": ChatGPTProvider}

# Cache provider instances for reuse (DEPRECATED: Use create_provider instead)
_PROVIDER_INSTANCES = {}
_provider_lock = threading.Lock()


def create_provider(provider_name):
    """Factory function to create a new provider instance

    Creates a fresh provider instance for session-scoped usage.
    Each call returns a new instance with isolated state (cache, clients).

    Args:
        provider_name: Name of the provider ('gemini', 'chatgpt', etc.)

    Returns:
        LLMProvider: New instance of the requested provider

    Raises:
        ValueError: If provider_name is not registered
    """
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    provider_class = _PROVIDERS[provider_name]
    return provider_class()


def get_provider(provider_name):
    """Factory function to get a provider instance (thread-safe)

    DEPRECATED: This function returns a global shared instance.
    New code should use create_provider() for session-scoped providers.

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


# ========================================
# Utility functions
# ========================================


def load_api_key(env_var_name: str) -> str:
    """Load API key from environment

    Args:
        env_var_name: Name of the environment variable (e.g., 'GOOGLE_API_KEY')

    Returns:
        str: API key value or None if not found
    """
    return os.getenv(env_var_name)
