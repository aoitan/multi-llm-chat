"""Runtime initialization for multi-llm-chat applications.

This module provides initialization functions for CLI and WebUI applications.
Call init_runtime() once at application startup before importing other
multi_llm_chat modules that depend on environment variables or configuration.
"""

import asyncio
import atexit
import logging
import os
import signal
import sys
import threading
from typing import TYPE_CHECKING, Optional

from dotenv import load_dotenv

from .config import load_config_from_env, set_config

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)
_initialized = False
_init_lock = threading.Lock()


def init_runtime(log_level: Optional[str] = None) -> None:
    """Initialize runtime environment for CLI/WebUI applications.

    This function:
    1. Loads environment variables from .env file
    2. Initializes the global configuration repository
    3. Optionally configures logging

    This should be called once at application startup before importing
    other multi_llm_chat modules that depend on configuration.

    Thread-safe: Uses double-checked locking to prevent race conditions
    in multi-threaded environments (e.g., WebUI with Gradio).

    Args:
        log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   If None, logging configuration is not modified.

    Note:
        This function is idempotent. Once initialized, subsequent calls
        are silently ignored, including log_level settings. To change
        log level after initialization, use logging.basicConfig() directly.

    Example:
        >>> from multi_llm_chat.runtime import init_runtime
        >>> init_runtime()  # Load .env and initialize config
        >>> # Now safe to import other modules
        >>> from multi_llm_chat.cli import main

    Raises:
        ValueError: If an invalid log_level is provided.
    """
    global _initialized

    # Fast path: already initialized (no lock needed)
    if _initialized:
        logger.debug("Runtime already initialized, skipping")
        return

    # Double-checked locking for thread safety
    with _init_lock:
        # Check again inside lock to prevent race condition
        if _initialized:
            logger.debug("Runtime already initialized (detected in lock), skipping")
            return

        try:
            # Load environment variables from .env file
            load_dotenv()

            # Load and set global configuration
            config = load_config_from_env()
            set_config(config)

            # Initialize MCP if enabled
            if config.mcp_enabled:
                from .mcp import get_mcp_manager

                # Skip if already initialized (idempotency)
                # Already inside _init_lock, no need for additional locking
                if get_mcp_manager() is None:
                    _init_mcp(config)
                else:
                    logger.debug("MCP manager already initialized, skipping")

            # Setup logging if specified
            if log_level:
                numeric_level = getattr(logging, log_level.upper(), None)
                if not isinstance(numeric_level, int):
                    raise ValueError(f"Invalid log level: {log_level}")
                logging.basicConfig(level=numeric_level)

            _initialized = True
            logger.debug("Runtime initialized successfully")
        except Exception:
            # Clean up partial initialization on error
            from .config import reset_config

            reset_config()
            raise


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


def _init_mcp(config: "AppConfig") -> None:
    """Initialize MCP server infrastructure.

    Args:
        config: AppConfig instance with MCP settings

    Raises:
        RuntimeError: If called from an async context with a running event loop.
    """
    from .mcp import MCPServerManager, reset_mcp_manager, set_mcp_manager
    from .mcp.filesystem_server import create_filesystem_server_config

    # Check if we're in an async context
    try:
        asyncio.get_running_loop()
        # If we reach here, there's a running loop (async context)
        raise RuntimeError(
            "_init_mcp() cannot be called from an async context. "
            "init_runtime() must be called before starting any event loop."
        )
    except RuntimeError as e:
        # Check if this is the expected "no running event loop" error
        error_msg = str(e).lower()
        if "no running event loop" not in error_msg and "no running loop" not in error_msg:
            # Unexpected RuntimeError - re-raise
            raise
        # No running loop - safe to proceed with asyncio.run()

    logger.info("Initializing MCP servers...")

    manager_set = False

    try:
        # Create manager
        manager = MCPServerManager()

        # Check for dangerous path override
        allow_dangerous_str = os.getenv("MCP_ALLOW_DANGEROUS_PATHS", "false").lower()
        allow_dangerous = allow_dangerous_str in ("true", "1", "yes")

        # Add filesystem server
        fs_config = create_filesystem_server_config(
            config.mcp_filesystem_root,
            timeout=config.mcp_timeout_seconds,
            allow_dangerous=allow_dangerous,
        )
        manager.add_server(fs_config)

        # Start all servers
        asyncio.run(manager.start_all())

        # Set global manager
        set_mcp_manager(manager)
        manager_set = True

        # Register cleanup handler
        def cleanup():
            logger.info("Stopping MCP servers...")
            # Try graceful shutdown first
            try:
                asyncio.run(manager.stop_all())
                logger.debug("MCP servers stopped gracefully")
                return
            except RuntimeError as e:
                # If we're in an async context at exit, fall back to force stop
                error_msg = str(e).lower()
                if "running" in error_msg and "loop" in error_msg:
                    logger.warning("Event loop active during cleanup, using force stop")
                else:
                    logger.warning(f"Could not cleanly stop MCP servers: {e}")

            # Fallback: forcefully terminate subprocesses
            try:
                manager.force_stop_all()
                logger.debug("MCP servers force stopped")
            except Exception as cleanup_error:
                logger.error(f"Failed to force stop MCP servers: {cleanup_error}")

        # Register cleanup handlers
        # Note: Signal handlers can only be registered from the main thread
        if threading.current_thread() is threading.main_thread():
            # Save existing handlers to chain them (avoid overriding app handlers)
            original_sigterm = signal.getsignal(signal.SIGTERM)
            original_sigint = signal.getsignal(signal.SIGINT)

            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, cleaning up MCP servers...")
                cleanup()

                # Chain to original handler if it was custom (not default)
                is_custom_sigterm = (
                    signum == signal.SIGTERM
                    and callable(original_sigterm)
                    and original_sigterm != signal.SIG_DFL
                )
                is_custom_sigint = (
                    signum == signal.SIGINT
                    and callable(original_sigint)
                    and original_sigint != signal.SIG_DFL
                )

                if is_custom_sigterm:
                    try:
                        original_sigterm(signum, frame)
                    except Exception as e:
                        logger.warning(f"Original SIGTERM handler raised: {e}")
                elif is_custom_sigint:
                    try:
                        original_sigint(signum, frame)
                    except Exception as e:
                        logger.warning(f"Original SIGINT handler raised: {e}")
                else:
                    # Default behavior: exit with proper signal exit code
                    # Convention: 128 + signal number (130 for SIGINT, 143 for SIGTERM)
                    sys.exit(128 + signum)

            # Register signal handlers (SIGTERM, SIGINT)
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            logger.debug("Signal handlers registered for SIGTERM and SIGINT")
        else:
            logger.info(
                "Skipping signal handler registration (not in main thread). "
                "MCP cleanup will only run via atexit."
            )

        # Always register atexit cleanup (works from any thread)
        atexit.register(cleanup)

        logger.info("MCP servers initialized successfully")
    except Exception:
        # Cleanup any partially started servers
        if manager_set:
            # Manager was successfully registered - use reset to clean up
            try:
                reset_mcp_manager()
            except Exception:
                # Swallow cleanup errors to avoid masking the original exception
                logger.exception("Failed to reset MCP manager after initialization error")
        elif "manager" in locals():
            # Manager was created but not registered - clean up directly
            try:
                asyncio.run(manager.stop_all())
            except Exception:
                # If graceful stop fails, force cleanup
                try:
                    manager.force_stop_all()
                except Exception:
                    logger.exception("Failed to cleanup partially started MCP servers")
        raise
