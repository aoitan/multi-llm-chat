"""
MCP server configuration data structures.

This module defines configuration classes for MCP servers.
"""

from dataclasses import dataclass


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.

    Attributes:
        name: Unique identifier for this server instance
        server_command: Command to launch the MCP server (e.g., "uvx", "python")
        server_args: Arguments for the server command (e.g., ["mcp-server-time"])
        timeout: Connection timeout in seconds (default: 10)
    """

    name: str
    server_command: str
    server_args: list[str]
    timeout: int = 10

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues.

        Returns:
            list[str]: List of validation error messages. Empty if valid.
        """
        issues = []

        if not self.name:
            issues.append("Server name cannot be empty")

        if not self.server_command:
            issues.append("Server command cannot be empty")

        if self.timeout <= 0:
            issues.append(f"Invalid timeout: {self.timeout} (must be > 0)")

        return issues
