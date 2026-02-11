"""Core module - Facade for multi-LLM chat functionality.

This module provides the public API for multi-LLM chat functionality:
- core_modules.token_and_context: Token calculation and validation
- core_modules.agentic_loop: Agentic Loop implementation
- core_modules.providers_facade: Provider access utilities
- llm_provider: Provider classes and configuration
- history_utils: History management utilities

⚠️ LEGACY API NOTICE:
Legacy API functions (call_*_api, format_history_*, stream_*, etc.) are still
available via direct import but are NOT part of the official public API.
These functions are deprecated and will be removed in v2.0.0.

To use legacy APIs, import directly from core_modules.legacy_api:
    from multi_llm_chat.core_modules.legacy_api import call_gemini_api

For migration guidance, see doc/deprecation_policy.md

Note: Environment variables should be loaded by calling init_runtime()
at application startup (see app.py, chat_logic.py).
"""

import logging

# Setup logger
logger = logging.getLogger(__name__)

# Import legacy API wrappers from core_modules (DEPRECATED functions)
# Import Agentic Loop implementation from core_modules
from .core_modules.agentic_loop import (  # noqa: F401
    AgenticLoopResult,
    execute_with_tools,
    execute_with_tools_stream,
    execute_with_tools_sync,
)
from .core_modules.legacy_api import (  # noqa: F401
    call_chatgpt_api,
    call_chatgpt_api_async,
    call_gemini_api,
    call_gemini_api_async,
    extract_text_from_chunk,
    format_history_for_chatgpt,
    format_history_for_gemini,
    load_api_key,
    prepare_request,
    stream_text_events,
    stream_text_events_async,
)

# Import provider facade from core_modules
from .core_modules.providers_facade import list_gemini_models  # noqa: F401

# Import token and context management wrappers from core_modules
from .core_modules.token_and_context import (  # noqa: F401
    calculate_tokens,
    get_max_context_length,
    get_pruning_info,
    get_token_info,
    prune_history_sliding_window,
    validate_context_length,
    validate_system_prompt_length,
)
from .history_utils import LLM_ROLES  # noqa: F401
from .llm_provider import (
    ChatGPTProvider as ChatGPTProvider,
)
from .llm_provider import (
    GeminiProvider as GeminiProvider,
)
from .llm_provider import (
    create_provider as create_provider,
)
from .llm_provider import (
    get_provider as get_provider,
)

# Define public API (Issue #115: Legacy APIs removed from __all__)
__all__ = [
    # Token and context management
    "calculate_tokens",
    "get_max_context_length",
    "get_pruning_info",
    "get_token_info",
    "prune_history_sliding_window",
    "validate_context_length",
    "validate_system_prompt_length",
    # Agentic Loop
    "AgenticLoopResult",
    "execute_with_tools",
    "execute_with_tools_stream",
    "execute_with_tools_sync",
    # Provider facade
    "list_gemini_models",
    # History utils
    "LLM_ROLES",
    # Provider classes and configuration
    "ChatGPTProvider",
    "GeminiProvider",
    "create_provider",
    "get_provider",
]


def __getattr__(name):
    """Lazy evaluation of deprecated constants for backward compatibility."""
    # Delegate to llm_provider's __getattr__ for deprecated constants
    from . import llm_provider

    if hasattr(llm_provider, "__getattr__"):
        try:
            return llm_provider.__getattr__(name)
        except AttributeError:
            pass
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
