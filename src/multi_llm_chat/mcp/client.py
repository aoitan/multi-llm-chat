"""
MCP client implementation for connecting to MCP servers.
"""

from contextlib import asynccontextmanager

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPClient:
    """Client for connecting to MCP servers via stdio."""

    def __init__(self, server_command, server_args):
        """Initialize MCPClient with server command and arguments.

        Args:
            server_command: Command to launch the MCP server (e.g., "uvx", "node")
            server_args: Arguments for the server command (e.g., ["mcp-server-time"])
        """
        self.server_command = server_command
        self.server_args = server_args

    @asynccontextmanager
    async def connect(self):
        """Connect to the MCP server and return a session.

        Yields:
            ClientSession: An initialized MCP client session

        Raises:
            ConnectionError: If server connection fails
            asyncio.TimeoutError: If connection times out
        """
        server_params = StdioServerParameters(command=self.server_command, args=self.server_args)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def list_tools(self, session):
        """List available tools from the connected MCP server.

        Args:
            session: An active ClientSession

        Returns:
            List of tool definitions with name, description, and inputSchema
        """
        response = await session.list_tools()
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
            for tool in response.tools
        ]
