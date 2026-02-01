"""Core sub-package for multi_llm_chat

This package contains the split implementation of core.py, organized by responsibility:

- agentic_loop: Agentic Loop execution with MCP/tool support
- legacy_api: DEPRECATED backward compatibility wrappers
- token_and_context: Token calculation and context validation wrappers
- providers_facade: Provider management and debug utilities

The parent core.py module re-exports all public APIs for backward compatibility.
"""

__all__ = []  # Public APIs are re-exported from parent core.py
