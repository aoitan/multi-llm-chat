"""Backward compatibility layer - delegates to chat_service

DEPRECATED: This module is deprecated. Import from chat_service instead:
    from multi_llm_chat.chat_service import ChatService, parse_mention, ASSISTANT_LABELS
"""

import warnings

# Import from new chat_service module
from .chat_service import ASSISTANT_LABELS, ChatService, parse_mention

# Re-export from core (non-legacy items)
from .core import (
    CHATGPT_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    list_gemini_models,
)

# Re-export legacy API from core_modules.legacy_api (Issue #115: explicit import)
from .core_modules.legacy_api import (
    call_chatgpt_api,
    call_gemini_api,
    format_history_for_chatgpt,
    format_history_for_gemini,
)
from .history import get_llm_response
from .history import reset_history as _reset_history

# Trigger deprecation warning on module import
warnings.warn(
    "The 'multi_llm_chat.chat_logic' module is deprecated. "
    "Import ChatService, parse_mention, and ASSISTANT_LABELS from "
    "'multi_llm_chat.chat_service' instead.",
    DeprecationWarning,
    stacklevel=2,
)


def main():
    """Backward compatible main function that returns only history"""
    import asyncio

    from .cli import main as _cli_main

    history, _system_prompt = asyncio.run(_cli_main())
    return history


def reset_history():
    """Clear conversation history (re-export for backward compatibility)."""
    return _reset_history()


__all__ = [
    "ChatService",
    "parse_mention",
    "ASSISTANT_LABELS",
    "main",
    "reset_history",
    "call_gemini_api",
    "call_chatgpt_api",
    "format_history_for_chatgpt",
    "format_history_for_gemini",
    "list_gemini_models",
    "get_llm_response",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_MODEL",
    "CHATGPT_MODEL",
]
