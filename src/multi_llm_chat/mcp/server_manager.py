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
        self._all_tools: list[dict] = []
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

        # Build tool-to-server mapping and cache all tools
        try:
            await self._build_tool_mapping()
            self._started = True
            logger.info("All MCP servers started successfully")
        except Exception:
            # Clean up any started servers if tool mapping fails
            await self.stop_all()
            raise

    async def _build_tool_mapping(self) -> None:
        """Build mapping from tool names to server names and cache all tools.

        Handles tool name conflicts by adding server name prefix.
        Fetches tools from all servers once and builds both the mapping and cached tool list.
        """
        self._tool_to_server.clear()
        self._all_tools.clear()
        tool_counts: dict[str, int] = {}
        server_tools: dict[str, list[dict]] = {}

        # Single pass: fetch tools from all servers
        for server_name, client in self._clients.items():
            tools = await client.list_tools()
            server_tools[server_name] = tools
            for tool in tools:
                tool_name = tool["name"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        # Build mapping and tool list with prefixes for conflicts
        for server_name, tools in server_tools.items():
            for tool in tools:
                original_name = tool["name"]
                # If tool name conflicts, add server prefix
                if tool_counts[original_name] > 1:
                    prefixed_name = f"{server_name}:{original_name}"
                    self._tool_to_server[prefixed_name] = server_name
                    # Add prefixed tool to cached list
                    prefixed_tool = tool.copy()
                    prefixed_tool["name"] = prefixed_name
                    self._all_tools.append(prefixed_tool)
                else:
                    self._tool_to_server[original_name] = server_name
                    # Add original tool to cached list
                    self._all_tools.append(tool)

    def get_all_tools(self) -> list[dict]:
        """Get aggregated tool list from all servers.

        Returns cached tool list built during start_all().

        Returns:
            list[dict]: List of tool definitions with name, description, inputSchema

        Raises:
            RuntimeError: If servers are not started
        """
        if not self._started:
            raise RuntimeError("Servers are not started. Call start_all() first.")

        return self._all_tools.copy()

    async def list_tools(self) -> list[dict]:
        """Alias for get_all_tools() to match MCPClient interface.

        This method exists for compatibility with the agentic loop which
        expects mcp_client.list_tools(). Internally delegates to get_all_tools().

        Returns:
            list[dict]: List of tool definitions with name, description, inputSchema

        Raises:
            RuntimeError: If servers are not started
        """
        return self.get_all_tools()

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
        original_tool_name = tool_name.split(":", 1)[-1]

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
        self._all_tools.clear()
        self._started = False
        logger.info("All MCP servers stopped")

    def force_stop_all(self) -> None:
        """Forcefully stop all running servers (synchronous).

        This method is intended for emergency cleanup (e.g., atexit handlers)
        when graceful async shutdown is not possible.

        Note: Since MCP client manages subprocess lifecycle internally via stdio_client,
        this method only clears internal state. Rely on __aexit__ for proper cleanup.
        """
        logger.warning(
            "force_stop_all() called - clearing state. "
            "Note: Subprocesses should be cleaned up via __aexit__"
        )

        self._clients.clear()
        self._tool_to_server.clear()
        self._all_tools.clear()
        self._started = False
        logger.info("All MCP servers force stopped")
