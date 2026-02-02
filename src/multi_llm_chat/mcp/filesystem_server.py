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
    - Paths that resolve to above via symlinks or relative paths
    - System directories themselves (/etc, /var, /tmp, etc. on POSIX - but not subdirectories)

    Note: Per spec and .env.example, subdirectories like /var/www/myproject are considered safe.

    Args:
        path: Path to check

    Returns:
        bool: True if path is dangerous, False otherwise
    """
    try:
        # Expand ~ and resolve to real path (handles symlinks and relative paths)
        resolved_path = Path(path).expanduser().resolve(strict=False)
    except (OSError, ValueError, RuntimeError) as e:
        # Path resolution failed - treat as dangerous (fail-safe)
        logger.warning(f"Could not resolve path '{path}': {e}, treating as dangerous")
        return True

    # Check for root directory (cross-platform)
    if resolved_path.parent == resolved_path:
        return True

    # Check for home directory
    try:
        home = Path.home().resolve()
        if resolved_path == home:
            return True
    except (OSError, RuntimeError):
        # Home directory unavailable - skip check
        pass

    # Check for system directories on POSIX systems
    # Only block the directories themselves, not subdirectories
    if os.name == "posix":
        system_dirs = [
            "/etc",
            "/bin",
            "/sbin",
            "/usr/bin",
            "/usr/sbin",
            "/boot",
            "/dev",
            "/proc",
            "/sys",
            "/tmp",
            "/var",
        ]
        for sys_dir in system_dirs:
            try:
                sys_path = Path(sys_dir).resolve()
                # Only block the system directory itself, not subdirectories
                # This allows /var/www/myproject, /tmp/build, etc.
                if resolved_path == sys_path:
                    return True
            except (ValueError, OSError):
                continue

    return False


def create_filesystem_server_config(
    root_dir: str | None = None, timeout: int = 120, allow_dangerous: bool = False
) -> MCPServerConfig:
    """Create configuration for filesystem MCP server.

    Args:
        root_dir: Root directory to expose. Defaults to current working directory.
        timeout: Connection timeout in seconds. Defaults to 120.
        allow_dangerous: Allow dangerous paths (/, home, system dirs). Defaults to False.

    Returns:
        MCPServerConfig: Configuration for filesystem server

    Raises:
        ValueError: If root_dir is dangerous and allow_dangerous=False, or if path is invalid.

    Security Note:
        Dangerous paths include:
        - Root directory (/, C:\\, etc.)
        - User home directory (~, /home/user, etc.)
        - System directories (/etc, /bin, /tmp, /var, etc.)

        To override, set environment variable MCP_ALLOW_DANGEROUS_PATHS=true
        or pass allow_dangerous=True (NOT RECOMMENDED).
    """
    if root_dir is None:
        root_dir = os.getcwd()

    # Expand ~ and resolve to absolute path
    try:
        resolved_root = Path(root_dir).expanduser().resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as e:
        raise ValueError(
            f"Invalid MCP_FILESYSTEM_ROOT: '{root_dir}' does not exist or is not accessible.\n"
            f"Error: {e}\n"
            f"Please set MCP_FILESYSTEM_ROOT to an existing directory."
        ) from e

    # Check if it's actually a directory
    if not resolved_root.is_dir():
        raise ValueError(
            f"Invalid MCP_FILESYSTEM_ROOT: '{root_dir}' is not a directory.\n"
            f"Please set MCP_FILESYSTEM_ROOT to a valid directory path."
        )

    # Check for dangerous paths
    if is_dangerous_path(str(resolved_root)):
        if not allow_dangerous:
            raise ValueError(
                f"[SECURITY ERROR] 🚨 Cannot expose dangerous path: {resolved_root}\n"
                f"This path contains sensitive system files.\n"
                f"\n"
                f"[DENIED] ❌ paths:\n"
                f"  - Root directory (/, C:\\, etc.)\n"
                f"  - Home directory (~, /home/user, /Users/user)\n"
                f"  - System directories (/etc, /bin, /tmp, /var, etc.)\n"
                f"\n"
                f"[SAFE] ✅ example: /path/to/specific/project/directory\n"
                f"\n"
                f"To override (NOT RECOMMENDED): Set MCP_ALLOW_DANGEROUS_PATHS=true"
            )
        else:
            # Log warning but allow
            logger.warning(
                f"[WARNING] ⚠️ DANGEROUS PATH ALLOWED: {resolved_root}\n"
                f"MCP filesystem server has read/write access to sensitive directories.\n"
                f"This is NOT RECOMMENDED for production use."
            )

    return MCPServerConfig(
        name="filesystem",
        server_command="uvx",
        server_args=["mcp-server-filesystem", str(resolved_root)],
        timeout=timeout,
    )
