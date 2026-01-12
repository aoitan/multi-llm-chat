"""
Tests for Agentic Loop implementation (execute_with_tools).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_execute_with_tools_single_iteration():
    """Test single tool call and final response."""
    from multi_llm_chat.core import execute_with_tools

    async def run_test():
        # Mock provider
        mock_provider = MagicMock()

        # First call: tool_call
        async def first_call(*args, **kwargs):
            yield {
                "type": "tool_call",
                "content": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            }

        # Second call: final text
        async def second_call(*args, **kwargs):
            yield {"type": "text", "content": "The weather in Tokyo is 25°C."}

        call_count = [0]

        async def mock_call_api(*args, **kwargs):
            if call_count[0] == 0:
                call_count[0] += 1
                async for item in first_call():
                    yield item
            else:
                async for item in second_call():
                    yield item

        mock_provider.call_api = mock_call_api

        # Mock MCP client
        mock_mcp = AsyncMock()
        mock_mcp.list_tools = AsyncMock(
            return_value=[{"name": "get_weather", "description": "Get weather", "inputSchema": {}}]
        )
        mock_mcp.call_tool = AsyncMock(
            return_value={"content": [{"type": "text", "text": "25°C"}], "isError": False}
        )

        # Execute
        history = []
        chunks = []
        async for chunk in execute_with_tools(mock_provider, history, mcp_client=mock_mcp):
            chunks.append(chunk)

        # Verify: tool_call → tool_result → text
        assert len(chunks) >= 3
        assert chunks[0]["type"] == "tool_call"
        assert chunks[1]["type"] == "tool_result"
        assert any(c["type"] == "text" for c in chunks)

        # Verify history structure
        assert len(history) == 2  # assistant (tool_call) + user (tool_result)
        assert history[0]["role"] == "assistant"
        assert history[0]["content"][0]["type"] == "tool_call"
        assert history[1]["role"] == "user"
        assert history[1]["content"][0]["type"] == "tool_result"

    asyncio.run(run_test())


def test_execute_with_tools_max_iterations():
    """Test loop termination at max_iterations."""
    from multi_llm_chat.core import execute_with_tools

    async def run_test():
        # Mock provider that always returns tool_call (infinite loop scenario)
        mock_provider = MagicMock()

        async def infinite_tool_call(*args, **kwargs):
            yield {
                "type": "tool_call",
                "content": {"name": "infinite_tool", "arguments": {}},
            }

        mock_provider.call_api = infinite_tool_call

        # Mock MCP client
        mock_mcp = AsyncMock()
        mock_mcp.list_tools = AsyncMock(
            return_value=[
                {"name": "infinite_tool", "description": "Never stops", "inputSchema": {}}
            ]
        )
        mock_mcp.call_tool = AsyncMock(
            return_value={"content": [{"type": "text", "text": "done"}], "isError": False}
        )

        # Execute with max_iterations=3
        history = []
        chunks = []
        async for chunk in execute_with_tools(
            mock_provider, history, mcp_client=mock_mcp, max_iterations=3
        ):
            chunks.append(chunk)

        # Should stop at 3 iterations
        tool_call_count = len([c for c in chunks if c["type"] == "tool_call"])
        assert tool_call_count == 3

    asyncio.run(run_test())


def test_execute_with_tools_timeout():
    """Test timeout handling."""
    from multi_llm_chat.core import execute_with_tools

    async def run_test():
        # Mock provider with slow response
        mock_provider = MagicMock()

        iteration_count = [0]

        async def slow_generator(*args, **kwargs):
            iteration_count[0] += 1
            # First iteration sleeps to trigger timeout
            if iteration_count[0] == 1:
                await asyncio.sleep(1.5)
            yield {"type": "text", "content": "done"}

        mock_provider.call_api = slow_generator

        # Mock MCP client
        mock_mcp = AsyncMock()
        mock_mcp.list_tools = AsyncMock(return_value=[])

        # Execute with short timeout
        with pytest.raises(TimeoutError):
            async for _ in execute_with_tools(mock_provider, [], mcp_client=mock_mcp, timeout=0.5):
                pass

    asyncio.run(run_test())


def test_execute_with_tools_tool_error():
    """Test graceful handling of tool execution errors."""
    from multi_llm_chat.core import execute_with_tools

    async def run_test():
        # Mock provider
        mock_provider = MagicMock()

        # First call: tool_call
        async def first_call(*args, **kwargs):
            yield {
                "type": "tool_call",
                "content": {"name": "failing_tool", "arguments": {}},
            }

        # Second call: final text (LLM acknowledges error)
        async def second_call(*args, **kwargs):
            yield {"type": "text", "content": "I encountered an error."}

        call_count = [0]

        async def mock_call_api(*args, **kwargs):
            if call_count[0] == 0:
                call_count[0] += 1
                async for item in first_call():
                    yield item
            else:
                async for item in second_call():
                    yield item

        mock_provider.call_api = mock_call_api

        # Mock MCP client that throws error
        mock_mcp = AsyncMock()
        mock_mcp.list_tools = AsyncMock(
            return_value=[{"name": "failing_tool", "description": "Fails", "inputSchema": {}}]
        )
        mock_mcp.call_tool = AsyncMock(side_effect=Exception("Tool crashed"))

        # Execute
        history = []
        chunks = []
        async for chunk in execute_with_tools(mock_provider, history, mcp_client=mock_mcp):
            chunks.append(chunk)

        # Should yield tool_result with error message
        tool_results = [c for c in chunks if c["type"] == "tool_result"]
        assert len(tool_results) == 1
        assert "Tool execution failed" in tool_results[0]["content"]["content"]

    asyncio.run(run_test())
