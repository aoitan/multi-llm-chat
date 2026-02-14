"""Backward compatibility layer - delegates to new webui module

DEPRECATED: This module is deprecated. Import from webui instead:
    from multi_llm_chat.webui import demo, launch, respond

Recommended usage: python -m multi_llm_chat.webui
"""

import warnings

# Trigger deprecation warning on module import (Issue #116)
warnings.warn(
    "The 'multi_llm_chat.app' module is deprecated. "
    "Import demo, launch, and respond from 'multi_llm_chat.webui' instead. "
    "Use 'python -m multi_llm_chat.webui' for launching the application.",
    DeprecationWarning,
    stacklevel=2,
)

from .webui import demo, launch, respond  # noqa: E402

__all__ = ["demo", "launch", "respond"]
