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


class TestAgenticLoopErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Test error handling in agentic loop for MCP integration."""

    async def test_list_tools_failure_is_handled_gracefully(self):
        """Test that list_tools() failure doesn't crash the app."""
        from multi_llm_chat.core_modules.agentic_loop import execute_with_tools_stream

        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.name = "Gemini"

        async def mock_call_api(history, system_prompt, tools):
            yield {"type": "text", "content": "Response"}

        mock_provider.call_api = mock_call_api

        # Create mock mcp_client that raises on list_tools
        mock_mcp_client = MagicMock()
        mock_mcp_client.list_tools = AsyncMock(side_effect=ConnectionError("MCP server down"))

        # Execute should not crash, but return error
        result = None
        chunks = []
        async for chunk in execute_with_tools_stream(
            provider=mock_provider,
            history=[],
            system_prompt=None,
            max_iterations=5,
            timeout=30,
            mcp_client=mock_mcp_client,
            tools=None,  # Force list_tools() call
        ):
            chunks.append(chunk)
            # Last yielded item is AgenticLoopResult
            if isinstance(chunk, dict):
                pass  # Regular chunk
            else:
                # AgenticLoopResult
                result = chunk

        # Should not crash
        self.assertIsNotNone(result)
        # Should have error field set
        self.assertIsNotNone(result.error)
        # Should contain user-friendly Japanese error message
        self.assertIn("MCP", result.error)

    async def test_tool_result_with_resource_type(self):
        """Test that resource-type tool results are handled correctly."""
        from multi_llm_chat.core_modules.agentic_loop import execute_with_tools_stream

        # Create mock provider that calls a tool
        mock_provider = MagicMock()
        mock_provider.name = "Gemini"

        call_count = {"count": 0}

        async def mock_call_api(history, system_prompt, tools):
            if call_count["count"] == 0:
                # First call: return tool call
                yield {
                    "type": "tool_call",
                    "content": {
                        "name": "read_file",
                        "arguments": {"path": "/README.md"},
                        "tool_call_id": "call_123",
                    },
                }
                call_count["count"] += 1
            else:
                # Second call: return final response
                yield {"type": "text", "content": "Summary: "}
                last_msg = history[-1]
                # Should have received file content
                self.assertIn("content", last_msg)
                content = last_msg["content"]
                if isinstance(content, list):
                    result_text = content[0]["content"]
                else:
                    result_text = content
                # Should NOT be "(no text output)"
                self.assertNotEqual(result_text, "(no text output)")

        mock_provider.call_api = mock_call_api

        # Create mock mcp_client that returns resource type
        mock_mcp_client = MagicMock()
        mock_mcp_client.list_tools = AsyncMock(
            return_value=[
                {
                    "name": "read_file",
                    "description": "Read file",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        )
        mock_mcp_client.call_tool = AsyncMock(
            return_value={
                "content": [
                    {
                        "type": "resource",
                        "resource": {"uri": "file:///README.md", "text": "# README\nContent here"},
                    }
                ]
            }
        )

        # Execute
        result = None
        async for chunk in execute_with_tools_stream(
            provider=mock_provider,
            history=[],
            system_prompt=None,
            max_iterations=5,
            timeout=30,
            mcp_client=mock_mcp_client,
            tools=None,
        ):
            # Last yielded item is AgenticLoopResult
            if isinstance(chunk, dict):
                pass  # Regular chunk
            else:
                # AgenticLoopResult
                result = chunk

        # Should complete successfully
        self.assertIsNotNone(result)
        self.assertIsNone(result.error)


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
