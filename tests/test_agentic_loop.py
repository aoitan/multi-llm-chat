"""
Tests for Agentic Loop implementation (execute_with_tools).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_execute_with_tools_single_iteration():
    """Test single tool call and final response."""
    from multi_llm_chat.core import execute_with_tools

    # Mock provider
    mock_provider = MagicMock()
    mock_provider.name = "gemini"

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
            async for chunk in first_call():
                yield chunk
        else:
            async for chunk in second_call():
                yield chunk

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
    result = await execute_with_tools(mock_provider, history, mcp_client=mock_mcp)
    chunks = result.chunks

    # Verify: tool_call → tool_result → text
    assert len(chunks) >= 3
    assert chunks[0]["type"] == "tool_call"
    assert chunks[1]["type"] == "tool_result"
    assert any(c["type"] == "text" for c in chunks)

    # Verify history structure (use history_delta)
    # assistant (tool_call) + tool (tool_result) + assistant (final text)
    assert len(result.history_delta) == 3
    assert result.history_delta[0]["role"] == "gemini"
    assert result.history_delta[0]["content"][0]["type"] == "tool_call"
    assert result.history_delta[0]["content"][0]["name"] == "get_weather"
    assert result.history_delta[1]["role"] == "tool"
    assert result.history_delta[1]["content"][0]["type"] == "tool_result"
    assert result.history_delta[2]["role"] == "gemini"
    assert result.history_delta[2]["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_execute_with_tools_max_iterations():
    """Test loop termination at max_iterations."""
    from multi_llm_chat.core import execute_with_tools

    # Mock provider that always returns tool_call (infinite loop scenario)
    mock_provider = MagicMock()
    mock_provider.name = "gemini"

    async def infinite_tool_call(*args, **kwargs):
        yield {
            "type": "tool_call",
            "content": {"name": "infinite_tool", "arguments": {}},
        }

    mock_provider.call_api = infinite_tool_call

    # Mock MCP client
    mock_mcp = AsyncMock()
    mock_mcp.list_tools = AsyncMock(
        return_value=[{"name": "infinite_tool", "description": "Never stops", "inputSchema": {}}]
    )
    mock_mcp.call_tool = AsyncMock(
        return_value={"content": [{"type": "text", "text": "done"}], "isError": False}
    )

    # Execute with max_iterations=3
    history = []
    result = await execute_with_tools(mock_provider, history, mcp_client=mock_mcp, max_iterations=3)
    chunks = result.chunks

    # Should stop at 3 iterations
    tool_call_count = len([c for c in chunks if c["type"] == "tool_call"])
    assert tool_call_count == 3
    assert result.iterations_used == 3


@pytest.mark.asyncio
async def test_execute_with_tools_timeout():
    """Test timeout handling."""
    from multi_llm_chat.core import execute_with_tools

    # Mock provider with slow response
    mock_provider = MagicMock()
    mock_provider.name = "gemini"

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
    result = await execute_with_tools(mock_provider, [], mcp_client=mock_mcp, timeout=0.5)

    # Should set timed_out flag instead of raising
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_execute_with_tools_tool_error():
    """Test graceful handling of tool execution errors."""
    from multi_llm_chat.core import execute_with_tools

    # Mock provider
    mock_provider = MagicMock()
    mock_provider.name = "gemini"

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
            async for chunk in first_call():
                yield chunk
        else:
            async for chunk in second_call():
                yield chunk

    mock_provider.call_api = mock_call_api

    # Mock MCP client that throws error
    mock_mcp = AsyncMock()
    mock_mcp.list_tools = AsyncMock(
        return_value=[{"name": "failing_tool", "description": "Fails", "inputSchema": {}}]
    )
    mock_mcp.call_tool = AsyncMock(side_effect=Exception("Tool crashed"))

    # Execute
    history = []
    result = await execute_with_tools(mock_provider, history, mcp_client=mock_mcp)
    chunks = result.chunks

    # Should yield tool_result with error message
    tool_results = [c for c in chunks if c["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert "ツール実行に失敗しました" in tool_results[0]["content"]["content"]


@pytest.mark.asyncio
async def test_execute_with_tools_missing_mcp_client():
    """Test graceful handling when tool call is made but mcp_client is missing."""
    from multi_llm_chat.core import execute_with_tools

    mock_provider = MagicMock()
    mock_provider.name = "gemini"

    async def mock_call_api(*args, **kwargs):
        yield {
            "type": "tool_call",
            "content": {"name": "any_tool", "arguments": {}},
        }

    mock_provider.call_api = mock_call_api

    history = []
    result = await execute_with_tools(mock_provider, history, mcp_client=None)
    chunks = result.chunks

    assert any("MCPクライアントが設定されていません" in str(c.get("content", "")) for c in chunks)

    # Verify history_delta has role 'tool' with error message
    assert any(h["role"] == "tool" for h in result.history_delta)
    error_msg = result.history_delta[-1]["content"][0]["content"]
    assert "MCPクライアントが設定されていません" in error_msg
