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
        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"], timeout=5)
        self.assertEqual(client.server_command, "uvx")
        self.assertEqual(client.server_args, ["mcp-server-time"])
        self.assertEqual(client.timeout, 5)

    @patch("asyncio.create_subprocess_exec")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_connect_success(self, mock_session_class, mock_create_subprocess):
        """MCPサーバーへの接続が成功する"""
        # Setup mocks
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_create_subprocess.return_value = mock_proc

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.initialize = AsyncMock()
        mock_session.close = AsyncMock()

        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"])

        async def run_test():
            async with client as connected_client:
                self.assertIsNotNone(connected_client.session)
                mock_session.initialize.assert_awaited_once()
            mock_session.close.assert_awaited_once()
            mock_proc.terminate.assert_called_once()

        asyncio.run(run_test())

    @patch("asyncio.create_subprocess_exec")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_list_tools(self, mock_session_class, mock_create_subprocess):
        """接続したサーバーからツール一覧を取得できる"""
        # Setup mocks
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_create_subprocess.return_value = mock_proc
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.initialize = AsyncMock()

        tool1 = MagicMock()
        tool1.name = "get_time"
        tool1.description = "Get current time"
        tool1.inputSchema = {"type": "object"}
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [tool1]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"])

        async def run_test():
            async with client as connected_client:
                tools = await connected_client.list_tools()
                self.assertEqual(len(tools), 1)
                self.assertEqual(tools[0]["name"], "get_time")

        asyncio.run(run_test())

    @patch("asyncio.create_subprocess_exec")
    def test_mcp_client_connection_error(self, mock_create_subprocess):
        """サーバー接続エラーが適切にハンドリングされる"""
        mock_create_subprocess.side_effect = FileNotFoundError("Command not found")

        client = MCPClient(server_command="invalid-command", server_args=[])

        async def run_test():
            with self.assertRaises(ConnectionError):
                async with client:
                    pass

        asyncio.run(run_test())

    @patch("asyncio.create_subprocess_exec")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_timeout(self, mock_session_class, mock_create_subprocess):
        """タイムアウトが適切にハンドリングされる"""
        # Setup mocks
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_create_subprocess.return_value = mock_proc

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.initialize = AsyncMock(side_effect=asyncio.TimeoutError("Connection timeout"))

        client = MCPClient(server_command="uvx", server_args=["slow-server"], timeout=0.1)

        async def run_test():
            with self.assertRaises(ConnectionError):
                async with client:
                    pass
            # Ensure cleanup is still attempted
            mock_proc.terminate.assert_called_once()


        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
