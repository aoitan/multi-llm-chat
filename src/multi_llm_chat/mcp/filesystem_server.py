"""Filesystem MCP server configuration and utilities.

This module provides utilities for configuring and managing the MCP filesystem server,
including dangerous path detection and configuration factory functions.
"""

import logging
import os
from pathlib import Path

from .server_config import MCPServerConfig

logger = logging.getLogger(__name__)


def is_dangerous_path(path: str) -> bool:
    """Check if a path is considered dangerous to expose.

    Dangerous paths include:
    - Root directory (/ on Unix, C:\\ on Windows, etc.)
    - User home directory

    Args:
        path: Path to check

    Returns:
        bool: True if path is dangerous, False otherwise
    """
    normalized = os.path.normpath(os.path.abspath(path))
    normalized_path = Path(normalized)

    # Check for root directory (cross-platform: POSIX /, Windows drive roots, UNC share roots)
    if normalized_path.parent == normalized_path:
        return True

    # Check for home directory
    home = os.path.normpath(os.path.abspath(str(Path.home())))
    if normalized == home:
        return True

    return False


def create_filesystem_server_config(
    root_dir: str | None = None, timeout: int = 120
) -> MCPServerConfig:
    """Create configuration for filesystem MCP server.

    Args:
        root_dir: Root directory to expose. Defaults to current working directory.
        timeout: Connection timeout in seconds. Defaults to 120.

    Returns:
        MCPServerConfig: Configuration for filesystem server

    Note:
        If root_dir is a dangerous path (e.g., / or home directory),
        a warning will be logged but the configuration will still be created.
    """
    if root_dir is None:
        root_dir = os.getcwd()

    # Warn if dangerous path
    if is_dangerous_path(root_dir):
        logger.warning(
            f"Filesystem server configured with dangerous path: {root_dir}. "
            "This exposes sensitive directories to the MCP server."
        )

    return MCPServerConfig(
        name="filesystem",
        server_command="uvx",
        server_args=["mcp-server-filesystem", root_dir],
        timeout=timeout,
    )
