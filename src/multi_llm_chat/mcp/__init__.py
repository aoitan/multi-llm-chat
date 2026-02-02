"""
MCP (Model Context Protocol) client package.
"""

from typing import Optional

from multi_llm_chat.mcp.client import MCPClient
from multi_llm_chat.mcp.server_manager import MCPServerManager

__all__ = ["MCPClient", "MCPServerManager"]


# Global MCP server manager instance
_mcp_manager: Optional[MCPServerManager] = None


def get_mcp_manager() -> Optional[MCPServerManager]:
    """Get the global MCP server manager instance.

    Returns:
        MCPServerManager or None: The global manager instance,
                                  or None if not initialized.
    """
    return _mcp_manager


def set_mcp_manager(manager: MCPServerManager) -> None:
    """Set the global MCP server manager instance.

    Args:
        manager: MCPServerManager instance to use globally.

    Raises:
        RuntimeError: If manager has already been set.
    """
    global _mcp_manager
    if _mcp_manager is not None:
        raise RuntimeError("MCP manager already set. Call reset_mcp_manager() first.")
    _mcp_manager = manager


def reset_mcp_manager() -> None:
    """Reset MCP manager state (for testing purposes only)."""
    global _mcp_manager
    _mcp_manager = None
