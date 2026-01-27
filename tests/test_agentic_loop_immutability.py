"""
Tests for AgenticLoopResult and execute_with_tools immutability.
"""

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_result_immutability():
    """AgenticLoopResult is immutable and provides all necessary data."""
    from multi_llm_chat.core import AgenticLoopResult

    result = AgenticLoopResult(
        chunks=[{"type": "text", "content": "hello"}],
        history_delta=[{"role": "assistant", "content": [{"type": "text", "content": "hello"}]}],
        final_text="hello",
        iterations_used=1,
        timed_out=False,
    )

    # Result should be frozen (dataclass with frozen=True)
    with pytest.raises(AttributeError):
        result.final_text = "modified"  # Should raise AttributeError


@pytest.mark.asyncio
async def test_history_not_mutated():
    """execute_with_tools does not mutate the original history."""
    from multi_llm_chat.core import execute_with_tools

    # Mock provider
    mock_provider = MagicMock()

    async def mock_response(*args, **kwargs):
        yield {"type": "text", "content": "Response"}

    mock_provider.call_api = mock_response

    # Mock MCP client
    mock_mcp = AsyncMock()
    mock_mcp.list_tools = AsyncMock(return_value=[])

    # Original history
    original_history = [{"role": "user", "content": [{"type": "text", "content": "Hello"}]}]
    history_copy_before = copy.deepcopy(original_history)

    # Execute
    result = await execute_with_tools(mock_provider, original_history, mcp_client=mock_mcp)

    # Original history should be unchanged
    assert original_history == history_copy_before
    assert len(result.history_delta) > 0
    assert result.final_text == "Response"


def test_execute_with_tools_sync_wrapper():
    """execute_with_tools_sync() provides sync interface."""
    from multi_llm_chat.core import execute_with_tools_sync

    # Mock provider
    mock_provider = MagicMock()
    mock_provider.name = "assistant"

    async def mock_call_api(*args, **kwargs):
        yield {"type": "text", "content": "Hello"}

    mock_provider.call_api = mock_call_api

    # Mock MCP client
    mock_mcp = MagicMock()
    mock_mcp.list_tools = AsyncMock(return_value=[])

    # Execute sync wrapper
    history = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    result = execute_with_tools_sync(mock_provider, history, mcp_client=mock_mcp)

    # Verify result
    assert result.final_text == "Hello"
    assert len(result.chunks) == 1
    assert result.chunks[0]["type"] == "text"


def test_sync_wrapper_raises_in_async_context():
    """execute_with_tools_sync() raises error if called from async context."""
    from multi_llm_chat.core import execute_with_tools_sync

    async def run_test():
        mock_provider = MagicMock()
        mock_provider.name = "assistant"

        async def mock_call_api(*args, **kwargs):
            yield {"type": "text", "content": "Hello"}

        mock_provider.call_api = mock_call_api

        mock_mcp = MagicMock()
        mock_mcp.list_tools = AsyncMock(return_value=[])

        # Should raise RuntimeError when called from async context
        with pytest.raises(RuntimeError, match="async context"):
            execute_with_tools_sync(mock_provider, [], mcp_client=mock_mcp)

    asyncio.run(run_test())


@pytest.mark.asyncio
async def test_history_delta_contains_only_new_entries():
    """history_delta contains only entries added during execution."""
    from multi_llm_chat.core import execute_with_tools

    # Mock provider
    mock_provider = MagicMock()
    mock_provider.name = "assistant"

    # First call: tool_call
    async def first_call(*args, **kwargs):
        yield {
            "type": "tool_call",
            "content": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
        }

    # Second call: final text
    async def second_call(*args, **kwargs):
        yield {"type": "text", "content": "Weather is 25°C"}

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

    # Original history with 2 entries
    original_history = [
        {"role": "user", "content": [{"type": "text", "content": "Previous message"}]},
        {"role": "assistant", "content": [{"type": "text", "content": "Previous response"}]},
    ]

    # Execute
    result = await execute_with_tools(mock_provider, original_history, mcp_client=mock_mcp)

    # history_delta should contain only new entries (not the original 2)
    assert len(result.history_delta) >= 2  # At least: assistant (tool_call) + tool (result)
    assert result.history_delta[0]["role"] in ["assistant", "tool"]
    assert result.final_text == "Weather is 25°C"


@pytest.mark.asyncio
async def test_provider_does_not_mutate_history():
    """Verify that Provider.call_api() respects the immutability contract."""
    from multi_llm_chat.core import execute_with_tools

    # Create a provider that attempts to mutate history
    mock_provider = MagicMock()
    mock_provider.name = "assistant"

    async def mutating_call_api(history, system_prompt=None, tools=None):
        # Simulate a buggy provider that tries to mutate history
        try:
            history.append({"role": "hacker", "content": "pwned"})
        except Exception:
            pass  # Mutation should fail if history is immutable
        yield {"type": "text", "content": "Hello"}

    mock_provider.call_api = mutating_call_api

    mock_mcp = MagicMock()
    mock_mcp.list_tools = AsyncMock(return_value=[])

    original_history = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    original_len = len(original_history)

    # Execute
    await execute_with_tools(mock_provider, original_history, mcp_client=mock_mcp)

    # Original history should remain unchanged
    assert len(original_history) == original_len
    assert original_history[0]["role"] == "user"
    # No "hacker" role should be present
    assert not any(entry.get("role") == "hacker" for entry in original_history)


def test_deepcopy_performance_within_tolerance():
    """Verify deepcopy overhead is acceptable for typical usage."""
    import time

    # 100エントリの大規模履歴を作成
    large_history = [
        {"role": "user", "content": [{"type": "text", "content": f"message {i}"}]}
        for i in range(100)
    ]

    # Baseline: 何もしない（変数参照のみ）
    start = time.perf_counter()
    for _ in range(100):  # Reduced iterations for more accurate measurement
        dummy = large_history
        _ = len(dummy)  # Force actual reference
    baseline = time.perf_counter() - start

    # Test: deepcopy
    start = time.perf_counter()
    for _ in range(100):
        _ = copy.deepcopy(large_history)
    deepcopy_time = time.perf_counter() - start

    # Simple sanity check: deepcopy should complete within reasonable time
    # For 100 iterations on 100-entry history, expect < 1 second on modern hardware
    assert deepcopy_time < 1.0, f"deepcopy too slow: {deepcopy_time:.3f}s for 100 iterations"

    # Also verify overhead is reasonable (allow up to 100x baseline since baseline is tiny)
    if baseline > 0:
        overhead_pct = (deepcopy_time - baseline) / baseline * 100
        # This is informational only - main check is absolute time above
        print(
            f"deepcopy overhead: {overhead_pct:.1f}% "
            f"(baseline={baseline:.6f}s, deepcopy={deepcopy_time:.6f}s)"
        )
