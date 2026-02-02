"""
MCP (Model Context Protocol) client package.
"""

import logging
import threading
from typing import Optional

from multi_llm_chat.mcp.client import MCPClient
from multi_llm_chat.mcp.server_manager import MCPServerManager

__all__ = [
    "MCPClient",
    "MCPServerManager",
    "get_mcp_manager",
    "set_mcp_manager",
    "reset_mcp_manager",
    "reset_mcp_manager_async",
]

logger = logging.getLogger(__name__)


# Global MCP server manager instance
_mcp_manager: Optional[MCPServerManager] = None
_mcp_manager_lock = threading.Lock()


def get_mcp_manager() -> Optional[MCPServerManager]:
    """Get the global MCP server manager instance.

    Returns:
        MCPServerManager or None: The global manager instance,
                                  or None if not initialized.
    """
    with _mcp_manager_lock:
        return _mcp_manager


def set_mcp_manager(manager: MCPServerManager) -> None:
    """Set the global MCP server manager instance.

    Args:
        manager: MCPServerManager instance to use globally.

    Raises:
        RuntimeError: If manager has already been set.
    """
    global _mcp_manager
    with _mcp_manager_lock:
        if _mcp_manager is not None:
            raise RuntimeError("MCP manager already set. Call reset_mcp_manager() first.")
        _mcp_manager = manager


def reset_mcp_manager() -> None:
    """Reset MCP manager state (for testing purposes only).

    This function stops all running MCP servers before clearing the global reference.
    Should only be used in tests or during application shutdown.

    Warning:
        This function uses asyncio.run() internally and cannot be called
        from an async context. Use reset_mcp_manager_async() instead.

    Raises:
        RuntimeError: If called from an async context (running event loop detected)
    """
    import asyncio

    global _mcp_manager
    with _mcp_manager_lock:
        if _mcp_manager is not None:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # If we reach here, there's a running loop
                raise RuntimeError(
                    "reset_mcp_manager() cannot be called from async context. "
                    "Use 'await reset_mcp_manager_async()' instead."
                )
            except RuntimeError as e:
                if "no running event loop" not in str(e).lower():
                    raise

            # Safe to use asyncio.run()
            try:
                asyncio.run(_mcp_manager.stop_all())
                logger.debug("MCP servers stopped during reset")
            except Exception:
                logger.exception("Failed to stop MCP servers during reset")
        _mcp_manager = None


async def reset_mcp_manager_async() -> None:
    """Async version of reset_mcp_manager() for use in async contexts.

    This function should be used when resetting the MCP manager from
    within an async function or event loop.
    """
    global _mcp_manager
    with _mcp_manager_lock:
        if _mcp_manager is not None:
            try:
                await _mcp_manager.stop_all()
                logger.debug("MCP servers stopped during async reset")
            except Exception:
                logger.exception("Failed to stop MCP servers during async reset")
        _mcp_manager = None
