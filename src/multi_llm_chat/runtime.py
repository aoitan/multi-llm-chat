"""Runtime initialization for multi-llm-chat applications.

This module provides initialization functions for CLI and WebUI applications.
Call init_runtime() once at application startup before importing other
multi_llm_chat modules that depend on environment variables.
"""

import logging
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
_initialized = False


def init_runtime(log_level: Optional[str] = None) -> None:
    """Initialize runtime environment for CLI/WebUI applications.

    This should be called once at application startup before importing
    other multi_llm_chat modules that depend on environment variables.

    Args:
        log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   If None, logging configuration is not modified.

    Example:
        >>> from multi_llm_chat.runtime import init_runtime
        >>> init_runtime()  # Load .env variables
        >>> # Now safe to import other modules
        >>> from multi_llm_chat.cli import main
    """
    global _initialized
    if _initialized:
        logger.debug("Runtime already initialized, skipping")
        return

    # Load environment variables from .env file
    load_dotenv()

    # Setup logging if specified
    if log_level:
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logging.basicConfig(level=numeric_level)

    _initialized = True
    logger.debug("Runtime initialized successfully")


def is_initialized() -> bool:
    """Check if runtime has been initialized.

    Returns:
        bool: True if init_runtime() has been called, False otherwise.
    """
    return _initialized


def reset_runtime() -> None:
    """Reset initialization state (for testing purposes only).

    This function should only be used in tests to reset the module state
    between test cases.
    """
    global _initialized
    _initialized = False
