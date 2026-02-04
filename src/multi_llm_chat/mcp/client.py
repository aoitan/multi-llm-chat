"""
MCP client implementation for connecting to MCP servers.
"""

import asyncio
import logging
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting to MCP servers via stdio."""

    def __init__(self, server_command, server_args, timeout=10):
        """Initialize MCPClient with server command and arguments.

        Args:
            server_command: Command to launch the MCP server (e.g., "uvx", "node")
            server_args: Arguments for the server command (e.g., ["mcp-server-time"])
            timeout: Connection timeout in seconds.
        """
        self.server_command = server_command
        self.server_args = server_args
        self.timeout = timeout
        self.session = None
        self._exit_stack = None
        self._process = None  # Track subprocess for force termination

    async def __aenter__(self):
        """Connect to the MCP server and initialize a session."""
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.server_command,
                args=self.server_args,
            )

            # Use AsyncExitStack to manage the stdio_client context
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            # Connect to server using MCP's stdio_client
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            # Create session with properly wrapped streams
            self.session = ClientSession(read_stream, write_stream)
            await asyncio.wait_for(self.session.initialize(), timeout=self.timeout)
            return self
        except Exception as e:
            # On any exception during initialization, ensure cleanup
            await self.__aexit__(None, None, None)
            # Wrap all exceptions in ConnectionError to signal connection failure
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the session and cleanup resources."""
        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass  # Ignore errors on cleanup
            self.session = None

        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass  # Ignore errors on cleanup
            self._exit_stack = None

        self._process = None

    def force_terminate(self):
        """Forcefully terminate the server subprocess (synchronous).

        This method is intended for emergency cleanup when graceful async shutdown
        is not possible (e.g., atexit handlers, signal handlers).

        WARNING: This method attempts to cleanup synchronously, which may not be
        fully reliable. Prefer using async __aexit__ when possible.

        Implementation note: Since stdio_client manages subprocess internally and
        doesn't expose the process handle, this method has limited capability.
        The best approach is to avoid reaching this code path by using proper
        async cleanup via runtime.py's improved cleanup() function.
        """
        logger.warning(
            "force_terminate() called - attempting emergency cleanup. "
            "This may leave resources in inconsistent state."
        )

        # Clear internal state
        self.session = None
        self._exit_stack = None
        self._process = None

        logger.info("MCPClient force terminated (state cleared)")

    async def list_tools(self):
        """List available tools from the connected MCP server.

        Returns:
            List of tool definitions with name, description, and inputSchema

        Raises:
            ConnectionError: if the client is not connected.
        """
        if not self.session:
            raise ConnectionError("Client is not connected. Use 'async with MCPClient(...)'.")

        response = await self.session.list_tools()
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool on the MCP server.

        Args:
            name: Tool name (e.g., "get_weather")
            arguments: Tool arguments as dict (e.g., {"location": "Tokyo"})

        Returns:
            Tool result with structure:
            {
                "content": [
                    {"type": "text", "text": "..."},
                    # or {"type": "image", "data": "...", "mimeType": "..."},
                    # or {"type": "resource", "resource": {...}}
                ],
                "isError": bool  # Optional, indicates execution failure
            }

        Raises:
            ConnectionError: If session is not initialized
        """
        if not self.session:
            raise ConnectionError("Client is not connected. Use 'async with MCPClient(...)'.")

        response = await self.session.call_tool(name, arguments)

        # Convert MCP CallToolResult to dict format
        content = []
        for item in response.content:
            # Extract all fields from the content item
            item_dict = {"type": item.type}
            # Use model_dump to get all fields, excluding 'type' since we already have it
            item_dict.update(item.model_dump(exclude={"type"}))
            content.append(item_dict)

        return {
            "content": content,
            "isError": getattr(response, "isError", False),
        }
