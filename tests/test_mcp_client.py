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

    @patch("multi_llm_chat.mcp.client.AsyncExitStack")
    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_connect_success(
        self, mock_session_class, mock_stdio_client, mock_exit_stack_class
    ):
        """MCPサーバーへの接続が成功する"""
        # Setup mocks
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_client.return_value = (mock_read_stream, mock_write_stream)

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        mock_exit_stack = AsyncMock()
        mock_exit_stack_class.return_value = mock_exit_stack
        mock_exit_stack.__aenter__ = AsyncMock(return_value=mock_exit_stack)
        mock_exit_stack.enter_async_context = AsyncMock(
            side_effect=[
                (mock_read_stream, mock_write_stream),  # stdio_client
                mock_session,  # ClientSession
            ]
        )

        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"])

        async def run_test():
            async with client as connected_client:
                self.assertIsNotNone(connected_client.session)

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.AsyncExitStack")
    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_list_tools(
        self, mock_session_class, mock_stdio_client, mock_exit_stack_class
    ):
        """接続したサーバーからツール一覧を取得できる"""
        # Setup mocks
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_client.return_value = (mock_read_stream, mock_write_stream)

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        tool1 = MagicMock()
        tool1.name = "get_time"
        tool1.description = "Get current time"
        tool1.inputSchema = {"type": "object"}
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [tool1]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)

        mock_exit_stack = AsyncMock()
        mock_exit_stack_class.return_value = mock_exit_stack
        mock_exit_stack.__aenter__ = AsyncMock(return_value=mock_exit_stack)
        mock_exit_stack.enter_async_context = AsyncMock(
            side_effect=[
                (mock_read_stream, mock_write_stream),  # stdio_client
                mock_session,  # ClientSession
            ]
        )

        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"])

        async def run_test():
            async with client as connected_client:
                tools = await connected_client.list_tools()
                self.assertEqual(len(tools), 1)
                self.assertEqual(tools[0]["name"], "get_time")

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.stdio_client")
    def test_mcp_client_connection_error(self, mock_stdio_client):
        """サーバー接続エラーが適切にハンドリングされる"""
        mock_stdio_client.return_value.__aenter__ = AsyncMock(
            side_effect=FileNotFoundError("Command not found")
        )
        mock_stdio_client.return_value.__aexit__ = AsyncMock()

        client = MCPClient(server_command="invalid-command", server_args=[])

        async def run_test():
            with self.assertRaises(ConnectionError):
                async with client:
                    pass

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.AsyncExitStack")
    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_timeout(self, mock_session_class, mock_stdio_client, mock_exit_stack_class):
        """タイムアウトが適切にハンドリングされる"""
        # Setup mocks
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_client.return_value = (mock_read_stream, mock_write_stream)

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        mock_exit_stack = AsyncMock()
        mock_exit_stack_class.return_value = mock_exit_stack
        mock_exit_stack.__aenter__ = AsyncMock(return_value=mock_exit_stack)
        # Simulate timeout during ClientSession context entry
        mock_exit_stack.enter_async_context = AsyncMock(
            side_effect=asyncio.TimeoutError("Connection timeout")
        )

        client = MCPClient(server_command="uvx", server_args=["slow-server"], timeout=0.1)

        async def run_test():
            with self.assertRaises(ConnectionError):
                async with client:
                    pass

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.AsyncExitStack")
    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_mcp_client_unexpected_error_on_connect(
        self, mock_session_class, mock_stdio_client, mock_exit_stack_class
    ):
        """予期せぬエラー発生時にクリーンアップが実行される"""
        # Setup mocks
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_client.return_value = (mock_read_stream, mock_write_stream)

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        mock_exit_stack = AsyncMock()
        mock_exit_stack_class.return_value = mock_exit_stack
        mock_exit_stack.__aenter__ = AsyncMock(return_value=mock_exit_stack)
        # Simulate an unexpected error during session initialization
        mock_exit_stack.enter_async_context = AsyncMock(side_effect=ValueError("Unexpected error"))

        client = MCPClient(server_command="uvx", server_args=["buggy-server"])

        async def run_test():
            with self.assertRaises(ConnectionError) as cm:
                async with client:
                    pass
            # Check that the original exception is wrapped
            self.assertIsInstance(cm.exception.__cause__, ValueError)

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_call_tool_success(self, mock_session_class, mock_stdio_client):
        """ツール実行が成功する"""
        # Setup mocks
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_client.return_value.__aenter__ = AsyncMock(
            return_value=(mock_read_stream, mock_write_stream)
        )
        mock_stdio_client.return_value.__aexit__ = AsyncMock()

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.initialize = AsyncMock()

        # Mock tool result
        mock_content_item = MagicMock()
        mock_content_item.type = "text"
        mock_content_item.model_dump = MagicMock(return_value={"text": "Tokyo weather: 25°C"})

        mock_tool_result = MagicMock()
        mock_tool_result.content = [mock_content_item]
        mock_tool_result.isError = False

        mock_session.call_tool = AsyncMock(return_value=mock_tool_result)

        client = MCPClient(server_command="uvx", server_args=["mcp-server-weather"])

        async def run_test():
            async with client as connected_client:
                result = await connected_client.call_tool("get_weather", {"location": "Tokyo"})

                self.assertIn("content", result)
                self.assertEqual(len(result["content"]), 1)
                self.assertEqual(result["content"][0]["type"], "text")
                self.assertIn("25°C", result["content"][0]["text"])
                self.assertFalse(result["isError"])

                mock_session.call_tool.assert_awaited_once_with(
                    "get_weather", {"location": "Tokyo"}
                )

        asyncio.run(run_test())

    @patch("multi_llm_chat.mcp.client.stdio_client")
    @patch("multi_llm_chat.mcp.client.ClientSession")
    def test_call_tool_error(self, mock_session_class, mock_stdio_client):
        """ツール実行がエラーを返す"""
        # Setup mocks
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_client.return_value.__aenter__ = AsyncMock(
            return_value=(mock_read_stream, mock_write_stream)
        )
        mock_stdio_client.return_value.__aexit__ = AsyncMock()

        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.initialize = AsyncMock()

        # Mock tool error result
        mock_content_item = MagicMock()
        mock_content_item.type = "text"
        mock_content_item.model_dump = MagicMock(return_value={"text": "API key missing"})

        mock_tool_result = MagicMock()
        mock_tool_result.content = [mock_content_item]
        mock_tool_result.isError = True

        mock_session.call_tool = AsyncMock(return_value=mock_tool_result)

        client = MCPClient(server_command="uvx", server_args=["mcp-server-weather"])

        async def run_test():
            async with client as connected_client:
                result = await connected_client.call_tool("get_weather", {"location": "Invalid"})

                self.assertTrue(result["isError"])
                self.assertIn("API key missing", result["content"][0]["text"])

        asyncio.run(run_test())

    def test_call_tool_without_connection(self):
        """接続前にツール実行を試みるとエラーになる"""
        client = MCPClient(server_command="uvx", server_args=["mcp-server-time"])

        async def run_test():
            with self.assertRaises(ConnectionError):
                await client.call_tool("get_time", {})

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
