"""
MCP server manager for managing multiple MCP servers.

This module provides centralized management of multiple MCP server instances,
including lifecycle management, tool aggregation, and tool execution routing.
"""

import logging

from multi_llm_chat.mcp.client import MCPClient
from multi_llm_chat.mcp.server_config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manager for multiple MCP server instances.

    This class handles the lifecycle of multiple MCP servers, aggregates their
    tools, and routes tool calls to the appropriate server.
    """

    def __init__(self):
        """Initialize the MCP server manager."""
        self._servers: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._tool_to_server: dict[str, str] = {}
        self._started: bool = False

    def add_server(self, config: MCPServerConfig) -> None:
        """Add a server configuration to the manager.

        Args:
            config: MCP server configuration

        Raises:
            ValueError: If a server with the same name is already registered
        """
        if config.name in self._servers:
            raise ValueError(f"Server '{config.name}' is already registered")

        issues = config.validate()
        if issues:
            raise ValueError(f"Invalid server configuration: {', '.join(issues)}")

        self._servers[config.name] = config
        logger.debug(f"Added server: {config.name}")

    async def start_all(self) -> None:
        """Start all registered servers.

        Raises:
            RuntimeError: If servers are already started
        """
        if self._started:
            raise RuntimeError("Servers are already started")

        logger.info(f"Starting {len(self._servers)} MCP server(s)...")

        # Start all servers
        for name, config in self._servers.items():
            try:
                client = MCPClient(
                    server_command=config.server_command,
                    server_args=config.server_args,
                    timeout=config.timeout,
                )
                # Enter async context
                await client.__aenter__()
                self._clients[name] = client
                logger.info(f"Started server: {name}")
            except Exception as e:
                logger.error(f"Failed to start server '{name}': {e}")
                # Clean up any started servers
                await self.stop_all()
                raise

        # Build tool-to-server mapping
        await self._build_tool_mapping()

        self._started = True
        logger.info("All MCP servers started successfully")

    async def _build_tool_mapping(self) -> None:
        """Build mapping from tool names to server names.

        Handles tool name conflicts by adding server name prefix.
        """
        self._tool_to_server.clear()
        tool_counts: dict[str, int] = {}

        # First pass: count occurrences of each tool name
        for _server_name, client in self._clients.items():
            tools = await client.list_tools()
            for tool in tools:
                tool_name = tool["name"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        # Second pass: build mapping with prefixes for conflicts
        for server_name, client in self._clients.items():
            tools = await client.list_tools()
            for tool in tools:
                original_name = tool["name"]
                # If tool name conflicts, add server prefix
                if tool_counts[original_name] > 1:
                    prefixed_name = f"{server_name}:{original_name}"
                    self._tool_to_server[prefixed_name] = server_name
                else:
                    self._tool_to_server[original_name] = server_name

    async def get_all_tools(self) -> list[dict]:
        """Get aggregated tool list from all servers.

        Returns:
            list[dict]: List of tool definitions with name, description, inputSchema

        Raises:
            RuntimeError: If servers are not started
        """
        if not self._started:
            raise RuntimeError("Servers are not started. Call start_all() first.")

        all_tools = []
        tool_counts: dict[str, int] = {}

        # First pass: count tool name occurrences
        temp_tools: dict[str, list[tuple[str, dict]]] = {}
        for server_name, client in self._clients.items():
            tools = await client.list_tools()
            for tool in tools:
                tool_name = tool["name"]
                if tool_name not in temp_tools:
                    temp_tools[tool_name] = []
                temp_tools[tool_name].append((server_name, tool))
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        # Second pass: add tools with prefixes for conflicts
        for tool_name, server_tools in temp_tools.items():
            if tool_counts[tool_name] > 1:
                # Conflict: add server prefix
                for server_name, tool in server_tools:
                    prefixed_tool = tool.copy()
                    prefixed_tool["name"] = f"{server_name}:{tool_name}"
                    all_tools.append(prefixed_tool)
            else:
                # No conflict: use original name
                _, tool = server_tools[0]
                all_tools.append(tool)

        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool on the appropriate server.

        Args:
            tool_name: Tool name (may include server prefix for conflicting names)
            arguments: Tool arguments as dict

        Returns:
            Tool result dict with 'content' and 'isError' keys

        Raises:
            RuntimeError: If servers are not started
            ValueError: If tool is not found
        """
        if not self._started:
            raise RuntimeError("Servers are not started. Call start_all() first.")

        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            raise ValueError(f"Tool '{tool_name}' not found in any server")

        client = self._clients[server_name]
        # Extract original tool name if prefixed
        original_tool_name = tool_name.split(":", 1)[-1] if ":" in tool_name else tool_name

        return await client.call_tool(original_tool_name, arguments)

    async def stop_all(self) -> None:
        """Stop all running servers."""
        logger.info("Stopping all MCP servers...")

        for name, client in list(self._clients.items()):
            try:
                await client.__aexit__(None, None, None)
                logger.info(f"Stopped server: {name}")
            except Exception as e:
                logger.warning(f"Error stopping server '{name}': {e}")

        self._clients.clear()
        self._tool_to_server.clear()
        self._started = False
        logger.info("All MCP servers stopped")
