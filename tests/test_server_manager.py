"""
Tests for MCP server manager implementation.
"""

import unittest
from unittest.mock import AsyncMock, patch

from multi_llm_chat.mcp.server_config import MCPServerConfig
from multi_llm_chat.mcp.server_manager import MCPServerManager


class TestMCPServerManager(unittest.IsolatedAsyncioTestCase):
    """Test MCPServerManager for managing multiple MCP servers."""

    def test_add_server(self):
        """サーバー設定を追加できる"""
        manager = MCPServerManager()
        config = MCPServerConfig(
            name="test-server",
            server_command="uvx",
            server_args=["mcp-server-time"],
            timeout=10,
        )

        manager.add_server(config)
        self.assertIn("test-server", manager._servers)

    def test_add_server_duplicate_name_raises_error(self):
        """重複するサーバー名でエラーが発生する"""
        manager = MCPServerManager()
        config1 = MCPServerConfig(name="test-server", server_command="uvx", server_args=["server1"])
        config2 = MCPServerConfig(name="test-server", server_command="uvx", server_args=["server2"])

        manager.add_server(config1)
        with self.assertRaises(ValueError) as cm:
            manager.add_server(config2)
        self.assertIn("already registered", str(cm.exception))

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    async def test_start_single_server(self, mock_client_class):
        """単一サーバーが起動できる"""
        manager = MCPServerManager()
        config = MCPServerConfig(
            name="test-server",
            server_command="uvx",
            server_args=["mcp-server-time"],
            timeout=10,
        )
        manager.add_server(config)

        # Mock MCPClient
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        await manager.start_all()
        self.assertTrue(manager._started)
        mock_client_class.assert_called_once_with(
            server_command="uvx", server_args=["mcp-server-time"], timeout=10
        )

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    async def test_start_multiple_servers(self, mock_client_class):
        """複数サーバーが同時に起動できる"""
        manager = MCPServerManager()
        config1 = MCPServerConfig(
            name="server1", server_command="uvx", server_args=["mcp-server-time"]
        )
        config2 = MCPServerConfig(
            name="server2", server_command="uvx", server_args=["mcp-server-weather"]
        )
        manager.add_server(config1)
        manager.add_server(config2)

        # Mock MCPClient
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        await manager.start_all()
        self.assertTrue(manager._started)
        self.assertEqual(mock_client_class.call_count, 2)

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    async def test_get_all_tools_aggregates(self, mock_client_class):
        """全サーバーのツールリストが統合される"""
        manager = MCPServerManager()
        config1 = MCPServerConfig(name="server1", server_command="uvx", server_args=["s1"])
        config2 = MCPServerConfig(name="server2", server_command="uvx", server_args=["s2"])
        manager.add_server(config1)
        manager.add_server(config2)

        # Mock MCPClient with different tools per server
        mock_client1 = AsyncMock()
        mock_client1.__aenter__ = AsyncMock(return_value=mock_client1)
        mock_client1.__aexit__ = AsyncMock()
        mock_client1.list_tools = AsyncMock(
            return_value=[
                {
                    "name": "get_time",
                    "description": "Get current time",
                    "inputSchema": {"type": "object"},
                }
            ]
        )

        mock_client2 = AsyncMock()
        mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
        mock_client2.__aexit__ = AsyncMock()
        mock_client2.list_tools = AsyncMock(
            return_value=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "inputSchema": {"type": "object"},
                }
            ]
        )

        mock_client_class.side_effect = [mock_client1, mock_client2]

        await manager.start_all()
        tools = manager.get_all_tools()

        self.assertEqual(len(tools), 2)
        tool_names = [t["name"] for t in tools]
        self.assertIn("get_time", tool_names)
        self.assertIn("get_weather", tool_names)

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    async def test_call_tool_routes_correctly(self, mock_client_class):
        """ツール呼び出しが正しいサーバーにルーティングされる"""
        manager = MCPServerManager()
        config = MCPServerConfig(name="test-server", server_command="uvx", server_args=["s1"])
        manager.add_server(config)

        # Mock MCPClient
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "get_time", "description": "Get time", "inputSchema": {}}]
        )
        mock_client.call_tool = AsyncMock(
            return_value={
                "content": [{"type": "text", "text": "2024-01-01 12:00:00"}],
                "isError": False,
            }
        )
        mock_client_class.return_value = mock_client

        await manager.start_all()
        result = await manager.call_tool("get_time", {})

        self.assertIn("content", result)
        self.assertEqual(result["content"][0]["text"], "2024-01-01 12:00:00")
        mock_client.call_tool.assert_awaited_once_with("get_time", {})

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    async def test_call_tool_unknown_tool_raises_error(self, mock_client_class):
        """未知のツール名でエラーが発生する"""
        manager = MCPServerManager()
        config = MCPServerConfig(name="test-server", server_command="uvx", server_args=["s1"])
        manager.add_server(config)

        # Mock MCPClient
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        await manager.start_all()
        with self.assertRaises(ValueError) as cm:
            await manager.call_tool("unknown_tool", {})
        self.assertIn("not found", str(cm.exception))

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    async def test_stop_all_cleans_up(self, mock_client_class):
        """全サーバーが停止される"""
        manager = MCPServerManager()
        config1 = MCPServerConfig(name="server1", server_command="uvx", server_args=["s1"])
        config2 = MCPServerConfig(name="server2", server_command="uvx", server_args=["s2"])
        manager.add_server(config1)
        manager.add_server(config2)

        # Mock MCPClient
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        await manager.start_all()
        self.assertTrue(manager._started)

        await manager.stop_all()
        self.assertFalse(manager._started)
        # __aexit__ should be called for each client
        self.assertEqual(mock_client.__aexit__.await_count, 2)

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    async def test_tool_name_conflict_handling(self, mock_client_class):
        """ツール名の衝突時にサーバー名プレフィックスが付与される"""
        manager = MCPServerManager()
        config1 = MCPServerConfig(name="server1", server_command="uvx", server_args=["s1"])
        config2 = MCPServerConfig(name="server2", server_command="uvx", server_args=["s2"])
        manager.add_server(config1)
        manager.add_server(config2)

        # Mock MCPClient with conflicting tool names
        mock_client1 = AsyncMock()
        mock_client1.__aenter__ = AsyncMock(return_value=mock_client1)
        mock_client1.__aexit__ = AsyncMock()
        mock_client1.list_tools = AsyncMock(
            return_value=[{"name": "read", "description": "Read from server1", "inputSchema": {}}]
        )

        mock_client2 = AsyncMock()
        mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
        mock_client2.__aexit__ = AsyncMock()
        mock_client2.list_tools = AsyncMock(
            return_value=[{"name": "read", "description": "Read from server2", "inputSchema": {}}]
        )

        mock_client_class.side_effect = [mock_client1, mock_client2]

        await manager.start_all()
        tools = manager.get_all_tools()

        self.assertEqual(len(tools), 2)
        tool_names = [t["name"] for t in tools]
        # Check that prefixes are added to conflicting tools
        self.assertIn("server1:read", tool_names)
        self.assertIn("server2:read", tool_names)

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    async def test_start_all_cleans_up_on_tool_mapping_failure(self, mock_client_class):
        """ツールマッピング失敗時にクリーンアップされる"""
        manager = MCPServerManager()
        config = MCPServerConfig(name="test-server", server_command="uvx", server_args=["s1"])
        manager.add_server(config)

        # Mock MCPClient that fails during list_tools
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(side_effect=RuntimeError("Tool listing failed"))
        mock_client_class.return_value = mock_client

        with self.assertRaises(RuntimeError) as cm:
            await manager.start_all()
        self.assertIn("Tool listing failed", str(cm.exception))

        # Verify cleanup was called
        self.assertFalse(manager._started)
        mock_client.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
