"""End-to-end tests for MCP integration with LLM."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from multi_llm_chat.chat_logic import ChatService
from multi_llm_chat.mcp import reset_mcp_manager_async


class TestMCPE2EIntegrationWithMocks(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for MCP filesystem integration using mocks."""

    async def test_chat_service_routes_tool_calls_to_manager(self):
        """Test that ChatService properly routes tool calls through MCPServerManager."""
        # Create mock manager
        mock_manager = MagicMock()
        mock_manager.get_all_tools = MagicMock(
            return_value=[
                {
                    "name": "read_file",
                    "description": "Read a file from the filesystem",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Path to file"}},
                        "required": ["path"],
                    },
                }
            ]
        )
        mock_manager.call_tool = AsyncMock(
            return_value={"content": [{"type": "text", "text": "File content: Hello World"}]}
        )

        # Add list_tools alias
        async def mock_list_tools():
            return mock_manager.get_all_tools()

        mock_manager.list_tools = mock_list_tools

        # Create mock provider that calls read_file tool
        mock_provider = MagicMock()
        mock_provider.name = "Gemini"

        call_count = [0]

        # Simulate LLM response with tool call
        async def mock_call_api(history, system_prompt, tools):
            call_count[0] += 1

            # Verify tools were passed
            self.assertIsNotNone(tools)
            self.assertGreater(len(tools), 0)

            if call_count[0] == 1:
                # First response: tool call
                yield {
                    "type": "tool_call",
                    "content": {
                        "name": "read_file",
                        "arguments": {"path": "/test.txt"},
                        "tool_call_id": "call_123",
                    },
                }
            else:
                # Second response: final text after seeing tool result
                yield {"type": "text", "content": "The file contains hello world"}

        mock_provider.call_api = mock_call_api

        # Create ChatService with our manager
        with patch("multi_llm_chat.chat_logic.get_mcp_manager", return_value=mock_manager):
            service = ChatService(gemini_provider=mock_provider)

            # Verify service has access to manager
            self.assertEqual(service.mcp_client, mock_manager)

            # Process a message
            chunks = []
            async for _display, _logic, chunk in service.process_message("@gemini Read test.txt"):
                chunks.append(chunk)

            # Verify tool was called exactly once
            mock_manager.call_tool.assert_called_once_with("read_file", {"path": "/test.txt"})

            # Verify we got both tool result and final text
            self.assertGreater(len(chunks), 0)
            tool_chunks = [c for c in chunks if c.get("type") == "tool_result"]
            text_chunks = [c for c in chunks if c.get("type") == "text"]
            self.assertEqual(len(tool_chunks), 1)  # One tool result
            self.assertGreater(len(text_chunks), 0)  # At least one text chunk

            # Verify we got tool_result chunks
            tool_results = [c for c in chunks if c.get("type") == "tool_result"]
            self.assertGreater(len(tool_results), 0, "No tool results in response")

            # Verify file content was in result
            result_content = tool_results[0].get("content", {}).get("content", "")
            self.assertIn("Hello World", result_content)

    async def test_multiple_tool_calls_in_sequence(self):
        """Test that multiple tool calls are handled correctly."""
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
                },
                {
                    "name": "list_directory",
                    "description": "List directory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            ]
        )

        # Mock different responses for different tools
        async def mock_call_tool(tool_name, arguments):
            if tool_name == "list_directory":
                return {"content": [{"type": "text", "text": "file1.txt\nfile2.txt"}]}
            elif tool_name == "read_file":
                return {"content": [{"type": "text", "text": "File contents here"}]}
            return {"content": [{"type": "text", "text": "Unknown tool"}]}

        mock_manager.call_tool = AsyncMock(side_effect=mock_call_tool)

        async def mock_list_tools():
            return mock_manager.get_all_tools()

        mock_manager.list_tools = mock_list_tools

        # Mock provider that calls multiple tools
        mock_provider = MagicMock()
        mock_provider.name = "Gemini"

        call_count = [0]

        async def mock_call_api(history, system_prompt, tools):
            call_count[0] += 1

            if call_count[0] == 1:
                # First call: list directory
                yield {
                    "type": "tool_call",
                    "content": {
                        "name": "list_directory",
                        "arguments": {"path": "/"},
                        "tool_call_id": "call_1",
                    },
                }
            elif call_count[0] == 2:
                # Second call: read file
                yield {
                    "type": "tool_call",
                    "content": {
                        "name": "read_file",
                        "arguments": {"path": "/file1.txt"},
                        "tool_call_id": "call_2",
                    },
                }
            else:
                # Final response
                yield {"type": "text", "content": "Done processing files"}

        mock_provider.call_api = mock_call_api

        with patch("multi_llm_chat.chat_logic.get_mcp_manager", return_value=mock_manager):
            service = ChatService(gemini_provider=mock_provider)

            # Process message
            async for _display, _logic, _chunk in service.process_message("@gemini List and read"):
                pass

            # Verify both tools were called
            self.assertEqual(mock_manager.call_tool.call_count, 2)
            calls = mock_manager.call_tool.call_args_list

            # First call: list_directory
            self.assertEqual(calls[0][0][0], "list_directory")
            # Second call: read_file
            self.assertEqual(calls[1][0][0], "read_file")


class TestMCPIntegrationWithoutServers(unittest.IsolatedAsyncioTestCase):
    """Test MCP integration without real servers."""

    async def test_chat_service_works_without_mcp_enabled(self):
        """Test that ChatService works when MCP is not enabled."""
        # Ensure no global manager
        await reset_mcp_manager_async()

        with patch("multi_llm_chat.chat_logic.get_mcp_manager", return_value=None):
            service = ChatService()

            # Service should work without MCP
            self.assertIsNone(service.mcp_client)

            # Create mock provider
            mock_provider = MagicMock()
            mock_provider.name = "Gemini"

            async def mock_call_api(history, system_prompt, tools):
                # tools should be None when no MCP
                self.assertIsNone(tools)
                yield {"type": "text", "content": "Response without MCP"}

            mock_provider.call_api = mock_call_api
            service._gemini_provider = mock_provider

            # Should work without MCP tools
            async for _display, _logic, _chunk in service.process_message("@gemini test"):
                pass

    async def test_tool_error_is_reported_to_llm(self):
        """Test that tool execution errors are properly reported."""
        mock_manager = MagicMock()
        mock_manager.get_all_tools = MagicMock(
            return_value=[{"name": "read_file", "description": "Read file", "inputSchema": {}}]
        )
        mock_manager.call_tool = AsyncMock(side_effect=Exception("File not found"))

        async def mock_list_tools():
            return mock_manager.get_all_tools()

        mock_manager.list_tools = mock_list_tools

        mock_provider = MagicMock()
        mock_provider.name = "Gemini"

        call_count = [0]

        async def mock_call_api(history, system_prompt, tools):
            call_count[0] += 1

            if call_count[0] == 1:
                # First response: tool call that will fail
                yield {
                    "type": "tool_call",
                    "content": {
                        "name": "read_file",
                        "arguments": {"path": "/nonexistent.txt"},
                        "tool_call_id": "call_1",
                    },
                }
            else:
                # Second response: acknowledge error
                yield {"type": "text", "content": "Sorry, could not read the file"}

        mock_provider.call_api = mock_call_api

        with patch("multi_llm_chat.chat_logic.get_mcp_manager", return_value=mock_manager):
            service = ChatService(gemini_provider=mock_provider)

            chunks = []
            async for _display, _logic, chunk in service.process_message("@gemini test"):
                chunks.append(chunk)

            # Should have tool_result with error
            tool_results = [c for c in chunks if c.get("type") == "tool_result"]
            self.assertGreater(len(tool_results), 0)

            # Error should be in the result
            result_text = tool_results[0].get("content", {}).get("content", "")
            self.assertIn("失敗", result_text)  # Japanese error message


class TestMCPToolFormats(unittest.TestCase):
    """Test MCP tool format compatibility."""

    def test_mcp_tool_schema_format(self):
        """Test that MCP tools use correct JSON Schema format."""
        from multi_llm_chat.mcp.filesystem_server import create_filesystem_server_config

        # Create config (without starting server)
        config = create_filesystem_server_config("/tmp", allow_dangerous=True)

        # Verify config structure
        self.assertEqual(config.name, "filesystem")
        self.assertEqual(config.server_command, "uvx")
        self.assertIn("mcp-server-filesystem", config.server_args)
        self.assertEqual(config.timeout, 120)  # Default timeout
