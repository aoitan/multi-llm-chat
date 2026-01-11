"""
Tests for MCP client implementation.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from multi_llm_chat.mcp.client import MCPClient


class TestMCPClient(unittest.TestCase):
    """Test MCPClient for connecting to MCP servers and listing tools."""

    def test_mcp_client_initialization(self):
        """MCPClientが正常に初期化できる"""
        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"])
        self.assertEqual(client.server_command, "uvx")
        self.assertEqual(client.server_args, ["mcp-server-time"])

    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_connect_success(self, mock_session_class, mock_stdio_client):
        """MCPサーバーへの接続が成功する"""
        # Setup mocks
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio_client.return_value.__aenter__.return_value = (mock_read, mock_write)

        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_session.initialize = AsyncMock()

        # Run test
        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"])

        async def run_test():
            async with client.connect() as session:
                self.assertIsNotNone(session)
                mock_session.initialize.assert_awaited_once()

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_list_tools(self, mock_session_class, mock_stdio_client):
        """接続したサーバーからツール一覧を取得できる"""
        # Setup mocks
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio_client.return_value.__aenter__.return_value = (mock_read, mock_write)

        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_session.initialize = AsyncMock()

        # Mock list_tools response
        tool1 = MagicMock()
        tool1.name = "get_time"
        tool1.description = "Get current time"
        tool1.inputSchema = {"type": "object"}

        tool2 = MagicMock()
        tool2.name = "add_numbers"
        tool2.description = "Add two numbers"
        tool2.inputSchema = {"type": "object"}

        mock_tools_response = MagicMock()
        mock_tools_response.tools = [tool1, tool2]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

        # Run test
        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"])

        async def run_test():
            async with client.connect() as session:
                tools = await client.list_tools(session)
                self.assertEqual(len(tools), 2)
                self.assertEqual(tools[0]["name"], "get_time")
                self.assertEqual(tools[0]["description"], "Get current time")
                self.assertEqual(tools[1]["name"], "add_numbers")

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.stdio_client")
    def test_mcp_client_connection_error(self, mock_stdio_client):
        """サーバー接続エラーが適切にハンドリングされる"""
        # Setup mock to raise exception
        mock_stdio_client.return_value.__aenter__.side_effect = ConnectionError("Server not found")

        # Run test
        client = MCPClient(server_command="invalid-command", server_args=[])

        async def run_test():
            with self.assertRaises(ConnectionError):
                async with client.connect():
                    pass

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_timeout(self, mock_session_class, mock_stdio_client):
        """タイムアウトが適切にハンドリングされる"""
        # Setup mocks
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio_client.return_value.__aenter__.return_value = (mock_read, mock_write)

        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_session.initialize = AsyncMock(side_effect=asyncio.TimeoutError("Connection timeout"))

        # Run test
        client = MCPClient(server_command="uvx", server_args=["slow-server"])

        async def run_test():
            with self.assertRaises(asyncio.TimeoutError):
                async with client.connect():
                    pass

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
