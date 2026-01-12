"""
MCP client implementation for connecting to MCP servers.
"""

import asyncio

from mcp.client.session import ClientSession


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
        self.proc = None
        self.session = None

    async def __aenter__(self):
        """Connect to the MCP server and initialize a session."""
        try:
            self.proc = await asyncio.create_subprocess_exec(
                self.server_command,
                *self.server_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )

            self.session = ClientSession(self.proc.stdout, self.proc.stdin)
            await asyncio.wait_for(self.session.initialize(), timeout=self.timeout)
            return self
        except Exception as e:
            # On any exception during initialization, ensure the process is cleaned up.
            await self.__aexit__(None, None, None)
            # Wrap all exceptions in ConnectionError to signal connection failure.
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the session and terminate the server process."""
        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass  # Ignore errors on cleanup
            self.session = None

        if self.proc:
            if self.proc.returncode is None:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    if self.proc.returncode is None:
                        self.proc.kill()
                        await self.proc.wait()

            # Safeguard: Explicitly close streams if they are still open,
            # which can happen if initialization fails before session takes ownership.
            if self.proc.stdin and not self.proc.stdin.is_closing():
                self.proc.stdin.close()
            if self.proc.stdout and not self.proc.stdout.is_closing():
                self.proc.stdout.close()

            self.proc = None

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
