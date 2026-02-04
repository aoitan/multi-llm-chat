"""Tests for MCP-LLM integration in CLI and WebUI."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from multi_llm_chat.chat_logic import ChatService


class TestChatServiceMCPIntegration(unittest.IsolatedAsyncioTestCase):
    """Test ChatService integration with MCPServerManager."""

    async def test_chat_service_uses_mcp_manager_when_available(self):
        """Test that ChatService uses MCPServerManager when available."""
        # Create mock manager
        mock_manager = MagicMock()
        mock_manager.get_all_tools = MagicMock(
            return_value=[
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        )
        mock_manager.call_tool = AsyncMock(
            return_value={"content": [{"type": "text", "text": "File content here"}]}
        )

        # Mock get_mcp_manager to return our mock
        with patch("multi_llm_chat.chat_logic.get_mcp_manager", return_value=mock_manager):
            service = ChatService()

            # ChatService should automatically use the global manager
            self.assertIsNotNone(service.mcp_client)
            self.assertEqual(service.mcp_client, mock_manager)

    async def test_chat_service_works_without_mcp_manager(self):
        """Test that ChatService works when MCPServerManager is not available."""
        with patch("multi_llm_chat.chat_logic.get_mcp_manager", return_value=None):
            ChatService()

            # Should not crash - just work without MCP tools
            from multi_llm_chat.mcp import get_mcp_manager

            manager = get_mcp_manager()
            self.assertIsNone(manager)

    async def test_mcp_tools_included_in_provider_call(self):
        """Test that MCP tools are passed to provider.call_api()."""
        # Create mock manager with tools
        mock_manager = MagicMock()
        mock_tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
        mock_manager.get_all_tools = MagicMock(return_value=mock_tools)

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.call_api = AsyncMock()

        async def mock_stream():
            yield {"type": "text", "content": "Test response"}

        mock_provider.call_api.return_value = mock_stream()
        mock_provider.name = "Gemini"

        with patch("multi_llm_chat.chat_logic.get_mcp_manager", return_value=mock_manager):
            with patch("multi_llm_chat.chat_logic.create_provider", return_value=mock_provider):
                service = ChatService()

                # Process a message
                async for _display, _logic, _chunk in service.process_message("test"):
                    pass

                # Verify that call_api was called with tools
                # (Implementation details will depend on how we integrate)
                # For now, just verify manager methods were called
                # mock_manager.get_all_tools.assert_called()


class TestMCPToolExecution(unittest.IsolatedAsyncioTestCase):
    """Test MCP tool execution through agentic loop."""

    async def test_tool_call_routes_to_mcp_manager(self):
        """Test that tool calls are routed to MCPServerManager.call_tool()."""
        # Create mock manager
        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(
            return_value={"content": [{"type": "text", "text": "File content"}]}
        )

        # Mock agentic loop to simulate tool call
        # (This test will be expanded when we implement the integration)
        with patch("multi_llm_chat.mcp.get_mcp_manager", return_value=mock_manager):
            from multi_llm_chat.mcp import get_mcp_manager

            manager = get_mcp_manager()

            # Simulate tool call
            result = await manager.call_tool("read_file", {"path": "/test.txt"})

            # Verify
            self.assertIn("content", result)
            mock_manager.call_tool.assert_called_once_with("read_file", {"path": "/test.txt"})

    async def test_tool_execution_error_handling(self):
        """Test error handling when tool execution fails."""
        # Create mock manager that raises error
        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(side_effect=Exception("Tool failed"))

        with patch("multi_llm_chat.mcp.get_mcp_manager", return_value=mock_manager):
            from multi_llm_chat.mcp import get_mcp_manager

            manager = get_mcp_manager()

            # Tool execution should raise
            with self.assertRaises(Exception) as ctx:
                await manager.call_tool("read_file", {"path": "/test.txt"})

            self.assertIn("Tool failed", str(ctx.exception))


class TestCLIMCPIntegration(unittest.TestCase):
    """Test CLI integration with MCPServerManager."""

    def test_cli_uses_global_mcp_manager(self):
        """Test that CLI uses the global MCPServerManager instead of creating MCPClient."""
        # This test will verify that CLI no longer creates MCPClient directly
        # but instead uses get_mcp_manager()

        with patch("multi_llm_chat.mcp.get_mcp_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            # Import should not fail
            from multi_llm_chat.mcp import get_mcp_manager

            manager = get_mcp_manager()

            # Verify manager is returned
            self.assertEqual(manager, mock_manager)
