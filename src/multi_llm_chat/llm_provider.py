"""LLM Provider abstraction layer using Strategy pattern

This module provides a unified interface for different LLM providers (Gemini, ChatGPT, etc.),
making it easy to add new providers without modifying existing code.

REFACTORING NOTE (Issue #101):
This module is being split into providers/ package. OpenAI-specific code has been moved to
providers/openai.py and Gemini-specific code to providers/gemini.py.
The classes are re-exported here for backward compatibility.

DEPRECATION WARNING:
Direct access to environment variables via module-level constants
(OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_MODEL, CHATGPT_MODEL)
is deprecated. Use config.get_config() instead.
"""

import logging
import os
import threading
import warnings

from .config import get_config
from .providers.base import LLMProvider
from .providers.gemini import (
    GeminiProvider,
    GeminiToolCallAssembler,
    _parse_tool_response_payload,
    mcp_tools_to_gemini_format,
)
from .providers.openai import (
    TIKTOKEN_AVAILABLE,
    ChatGPTProvider,
    OpenAIToolCallAssembler,
    mcp_tools_to_openai_format,
    parse_openai_tool_call,
)

logger = logging.getLogger(__name__)


# Mapping of deprecated constants to their config attribute names
# Format: constant_name -> (config_attr, replacement_code)
DEPRECATED_CONSTANTS = {
    "OPENAI_API_KEY": ("openai_api_key", "config.get_config().openai_api_key"),
    "GOOGLE_API_KEY": ("google_api_key", "config.get_config().google_api_key"),
    "GEMINI_MODEL": ("gemini_model", "config.get_config().gemini_model"),
    "CHATGPT_MODEL": ("chatgpt_model", "config.get_config().chatgpt_model"),
    "MCP_ENABLED": ("mcp_enabled", "config.get_config().mcp_enabled"),
}


def __getattr__(name):
    """Lazy evaluation of deprecated environment variables for backward compatibility.

    This allows old code to access OPENAI_API_KEY, GOOGLE_API_KEY, etc.
    while internally using the new ConfigRepository pattern.

    Deprecated constants:
    - OPENAI_API_KEY, GOOGLE_API_KEY: Use get_config().openai_api_key/google_api_key
    - GEMINI_MODEL, CHATGPT_MODEL: Use get_config().gemini_model/chatgpt_model
    - MCP_ENABLED: Use get_config().mcp_enabled
    """
    if name in DEPRECATED_CONSTANTS:
        attr_name, replacement = DEPRECATED_CONSTANTS[name]
        warnings.warn(
            f"{name} constant is deprecated. Use {replacement}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(get_config(), attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
    "TIKTOKEN_AVAILABLE",
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
    # Deprecated in v1.X, will be removed in v2.0.0 (Issue #116)
    warnings.warn(
        "get_provider() is deprecated. Use create_provider() for session-scoped providers. "
        "Will be removed in v2.0.0",
        DeprecationWarning,
        stacklevel=2,
    )

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
