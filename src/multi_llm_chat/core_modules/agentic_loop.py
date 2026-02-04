"""Agentic Loop implementation module

This module contains the core Agentic Loop execution logic with MCP/tool support.
The Agentic Loop repeatedly calls an LLM and executes tools until:
- LLM returns text without tool calls
- max_iterations is reached
- timeout is exceeded
- An error occurs

Main components:
- AgenticLoopResult: Immutable result data structure
- execute_with_tools_stream: Streaming version (yields chunks in real-time)
- execute_with_tools: Collect-all version (returns final result)
- execute_with_tools_sync: Synchronous wrapper

For new code, prefer execute_with_tools_stream for streaming scenarios
or execute_with_tools for batch processing.
"""

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..history_utils import validate_history_entry as _validate_history_entry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgenticLoopResult:
    """Result of execute_with_tools() execution.

    This is an immutable data structure that contains all information
    about the Agentic Loop execution without side effects.

    Attributes:
        chunks: List of streaming chunks emitted during execution
        history_delta: New history entries to append (caller's responsibility)
        final_text: Final text response from LLM
        iterations_used: Number of iterations actually used
        timed_out: Whether execution was terminated due to timeout
        error: Error message if execution failed (None on success)
    """

    chunks: List[Dict[str, Any]]
    history_delta: List[Dict[str, Any]]
    final_text: str
    iterations_used: int
    timed_out: bool
    error: Optional[str] = None


async def execute_with_tools_stream(
    provider: Any,
    history: List[Dict],
    system_prompt: Optional[str] = None,
    mcp_client: Optional[Any] = None,
    max_iterations: int = 10,
    timeout: float = 120.0,
    tools: Optional[List[Dict[str, Any]]] = None,
):
    """Execute LLM call with Agentic Loop, streaming chunks in real-time.

    This is the recommended API for streaming scenarios (CLI, WebUI).
    Yields chunks as they are generated, then yields the final result.

    Repeatedly calls LLM and executes tools until:
    - LLM returns text without tool calls
    - max_iterations is reached
    - timeout is exceeded
    - An error occurs

    Args:
        provider: LLM provider instance (Gemini/ChatGPT)
        history: Conversation history (read-only, will NOT be mutated)
        system_prompt: Optional system prompt
        mcp_client: Optional MCP client for tool execution
        max_iterations: Maximum number of LLM calls (default: 10)
        timeout: Maximum total execution time in seconds (default: 120)
        tools: Optional tool definitions (JSON Schema format).
               If None, tools will be fetched from mcp_client.

    Yields:
        - Dict[str, Any]: Streaming chunks with type "text", "tool_call", or "tool_result"
        - AgenticLoopResult: Final result (yielded last)

    Raises:
        TimeoutError: If execution exceeds timeout
        ValueError: If tool call is requested but mcp_client is None
    """
    start_time = time.time()

    # Create working copy of history (do not mutate original)
    working_history = copy.deepcopy(history)
    original_history_length = len(history)
    chunks = []
    final_text = ""
    timed_out = False
    error = None

    iterations_count = 0
    try:
        # Merge explicit tools with MCP tools (Issue #84 PR#3)
        # This is inside try block to handle MCP connection failures gracefully
        mcp_tools = []
        if mcp_client:
            try:
                elapsed = time.time() - start_time
                remaining_timeout = max(0.1, timeout - elapsed)
                mcp_tools = await asyncio.wait_for(
                    mcp_client.list_tools(), timeout=remaining_timeout
                )
            except (ConnectionError, TimeoutError, Exception) as e:
                # MCP server is unavailable - fail gracefully without crashing
                logger.error("Failed to list MCP tools: %s", e)
                error_msg = f"MCPツールの取得に失敗しました: {str(e)}"
                error_chunk = {"type": "error", "content": error_msg}
                chunks.append(error_chunk)
                yield error_chunk
                # Set error and return result without raising
                error = error_msg
                # Yield final error result
                yield AgenticLoopResult(
                    chunks=chunks,
                    history_delta=[],
                    final_text="",
                    iterations_used=0,
                    timed_out=False,
                    error=error,
                )
                return  # Exit gracefully

        # Merge explicit tools with MCP tools
        # Explicit tools take precedence if provided
        tools = (tools or []) + mcp_tools

        for _iteration_index in range(max_iterations):
            # Check timeout before each iteration
            if time.time() - start_time > timeout:
                logger.warning("Agentic loop timed out after %.1f seconds", timeout)
                raise TimeoutError(f"Execution exceeded timeout of {timeout} seconds")

            # Call LLM
            tool_calls_in_turn = []
            thought_text = ""
            async for chunk in provider.call_api(working_history, system_prompt, tools=tools):
                # Check timeout during streaming
                if time.time() - start_time > timeout:
                    logger.warning("Agentic loop timed out during streaming")
                    raise TimeoutError(f"Execution exceeded timeout of {timeout} seconds")

                chunk_type = chunk.get("type")

                if chunk_type == "text":
                    content = chunk.get("content", "")
                    thought_text += content
                    chunks.append(chunk)
                    yield chunk  # ← Real-time streaming
                elif chunk_type == "tool_call":
                    tool_call = chunk.get("content", {})
                    tool_calls_in_turn.append(tool_call)
                    chunks.append(chunk)
                    yield chunk  # ← Real-time streaming

            # Count iteration after LLM call completes
            iterations_count += 1

            # If no tool calls, loop ends (final response)
            if not tool_calls_in_turn:
                # Add final text to working_history if it exists
                if thought_text:
                    working_history.append(
                        {
                            "role": provider.name,
                            "content": [{"type": "text", "content": thought_text}],
                        }
                    )
                    final_text = thought_text
                break

            # Tool calls received - execute them
            if tool_calls_in_turn and not mcp_client:
                error_text = (
                    "[System: ツール呼び出しが要求されましたが、"
                    "MCPクライアントが設定されていません。]"
                )
                logger.error("Tool call received but mcp_client is None")
                error = error_text

                # Append error to working_history
                working_history.append(
                    {
                        "role": provider.name,
                        "content": [
                            {"type": "text", "content": thought_text if thought_text else ""},
                            *[{"type": "tool_call", **tc} for tc in tool_calls_in_turn],
                        ],
                    }
                )
                tool_error_entry = {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "name": tc.get("name"),
                            "content": error_text,
                            # tool_call_id is required; use fallback if missing (defensive)
                            "tool_call_id": tc.get("tool_call_id") or tc.get("name") or "unknown",
                        }
                        for tc in tool_calls_in_turn
                    ],
                }
                # Warn if any tool_call_id is missing
                for tc in tool_calls_in_turn:
                    if not tc.get("tool_call_id"):
                        logger.warning(
                            "Tool call missing tool_call_id during timeout; "
                            "using fallback (name=%s)",
                            tc.get("name"),
                        )
                _validate_history_entry(tool_error_entry)
                working_history.append(tool_error_entry)
                error_chunk = {"type": "text", "content": f"\n{error_text}"}
                chunks.append(error_chunk)
                yield error_chunk  # ← Real-time streaming
                break

            # Execute tools and collect results
            tool_results = []
            for tool_call in tool_calls_in_turn:
                # Check timeout before each tool execution
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.warning("Agentic loop timed out before tool execution")
                    raise TimeoutError(f"Execution exceeded timeout of {timeout} seconds")

                name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                tool_call_id = tool_call.get("tool_call_id")  # OpenAI only

                try:
                    # Calculate remaining timeout for this tool
                    remaining_timeout = max(0.1, timeout - elapsed)  # Minimum 0.1s

                    result = await asyncio.wait_for(
                        mcp_client.call_tool(name, arguments), timeout=remaining_timeout
                    )

                    # Extract text content with fallback for non-text types (resource, image)
                    text_parts = []
                    for item in result.get("content", []):
                        item_type = item.get("type")

                        if item_type == "text":
                            # Primary: text type
                            text_parts.append(item.get("text", ""))
                        elif item_type == "resource":
                            # Fallback: extract from resource
                            resource = item.get("resource", {})
                            # Try to get text content from resource
                            if "text" in resource:
                                text_parts.append(resource["text"])
                            elif "uri" in resource:
                                # Fallback to URI representation
                                uri = resource["uri"]
                                mime_type = resource.get("mimeType", "unknown")
                                text_parts.append(f"[Resource: {uri} ({mime_type})]")
                                logger.debug(
                                    "Non-text resource converted to URI: type=%s, uri=%s",
                                    mime_type,
                                    uri,
                                )
                        elif item_type == "image":
                            # Fallback: placeholder for image
                            mime_type = item.get("mimeType", "image")
                            text_parts.append(f"[画像: {mime_type}]")
                            logger.debug(
                                "Image content converted to placeholder: type=%s", mime_type
                            )
                        else:
                            # Last resort: stringify the entire item
                            import json

                            text_parts.append(f"[Unknown content: {json.dumps(item)}]")
                            logger.debug("Unknown content type stringified: type=%s", item_type)

                    result_text = "\n".join(text_parts) if text_parts else "(no text output)"

                    # Check for error flag in MCP response
                    is_error = result.get("isError", False)
                    if is_error:
                        result_text = f"[ERROR] {result_text}"
                        logger.warning(f"Tool '{name}' returned error: {result_text}")

                    tool_result = {
                        "name": name,
                        "content": result_text,
                    }
                    if tool_call_id:
                        tool_result["tool_call_id"] = tool_call_id

                    tool_results.append(tool_result)
                    tool_result_chunk = {"type": "tool_result", "content": tool_result}
                    chunks.append(tool_result_chunk)
                    yield tool_result_chunk  # ← Real-time streaming

                except ConnectionError:
                    # MCP server is down - cannot continue
                    logger.error("MCP connection error during tool execution: %s", name)
                    raise
                except Exception as e:
                    # Tool execution failed - report to LLM
                    logger.warning("Tool execution failed for %s: %s", name, e)
                    error_text = f"ツール実行に失敗しました: {str(e)}"
                    tool_result = {
                        "name": name,
                        "content": error_text,
                    }
                    if tool_call_id:
                        tool_result["tool_call_id"] = tool_call_id

                    tool_results.append(tool_result)
                    tool_result_chunk = {"type": "tool_result", "content": tool_result}
                    chunks.append(tool_result_chunk)
                    yield tool_result_chunk  # ← Real-time streaming

            # Append tool_call and tool_result to working_history
            assistant_entry = {"role": provider.name, "content": []}
            if thought_text:
                assistant_entry["content"].append({"type": "text", "content": thought_text})
            for tc in tool_calls_in_turn:
                assistant_entry["content"].append({"type": "tool_call", **tc})
            working_history.append(assistant_entry)

            # Tool results as separate message
            tool_entry = {
                "role": "tool",
                "content": [],
            }
            for tr in tool_results:
                if not tr.get("tool_call_id"):
                    logger.warning(
                        "Tool result missing tool_call_id; using fallback (name=%s)", tr.get("name")
                    )
                tool_result_item = {
                    "type": "tool_result",
                    "name": tr.get("name"),
                    "content": tr["content"],
                    # tool_call_id is required; fallback to name if missing (defensive)
                    "tool_call_id": tr.get("tool_call_id") or tr.get("name") or "unknown",
                }
                tool_entry["content"].append(tool_result_item)
            _validate_history_entry(tool_entry)
            working_history.append(tool_entry)

        else:
            # Loop reached max_iterations without breaking
            logger.info("Agentic loop reached max_iterations=%d", max_iterations)

    except TimeoutError:
        timed_out = True
        logger.warning("Agentic loop timed out (timed_out=%s)", timed_out)
        raise
    except Exception as e:
        error = str(e)
        logger.exception("Unhandled error during agentic loop execution: %s", e)
        raise

    # Calculate history_delta (only new entries)
    history_delta = working_history[original_history_length:]

    # Yield final result
    yield AgenticLoopResult(
        chunks=chunks,
        history_delta=history_delta,
        final_text=final_text,
        iterations_used=iterations_count,
        timed_out=timed_out,
        error=error,
    )


async def execute_with_tools(
    provider: Any,
    history: List[Dict],
    system_prompt: Optional[str] = None,
    mcp_client: Optional[Any] = None,
    max_iterations: int = 10,
    timeout: float = 120.0,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AgenticLoopResult:
    """Execute LLM call with Agentic Loop for tool execution (buffered version).

    NOTE: This is a legacy API that buffers all chunks before returning.
    For streaming scenarios, use execute_with_tools_stream() instead.

    Repeatedly calls LLM and executes tools until:
    - LLM returns text without tool calls
    - max_iterations is reached
    - timeout is exceeded
    - An error occurs

    Args:
        provider: LLM provider instance (Gemini/ChatGPT)
        history: Conversation history (read-only, will NOT be mutated)
        system_prompt: Optional system prompt
        mcp_client: Optional MCP client for tool execution
        max_iterations: Maximum number of LLM calls (default: 10)
        timeout: Maximum total execution time in seconds (default: 120)
        tools: Optional tool definitions (JSON Schema format).
               If None, tools will be fetched from mcp_client.

    Returns:
        AgenticLoopResult: Immutable result object containing:
            - chunks: All streaming chunks
            - history_delta: New entries to append to history
            - final_text: Final text response
            - iterations_used: Number of iterations used
            - timed_out: Whether execution timed out

    Raises:
        TimeoutError: If execution exceeds timeout
        ValueError: If tool call is requested but mcp_client is None
    """
    # Collect all items from stream
    result = None
    async for item in execute_with_tools_stream(
        provider, history, system_prompt, mcp_client, max_iterations, timeout, tools
    ):
        if isinstance(item, AgenticLoopResult):
            result = item
    return result


def execute_with_tools_sync(
    provider,
    history: List[Dict],
    system_prompt: Optional[str] = None,
    mcp_client=None,
    max_iterations: int = 5,
    timeout: float = 120.0,
) -> AgenticLoopResult:
    """
    Synchronous wrapper for execute_with_tools().

    Args:
        provider: LLM provider instance
        history: Conversation history (will be deep-copied, not mutated)
        system_prompt: Optional system prompt
        mcp_client: MCP client for tool execution
        max_iterations: Maximum tool loop iterations
        timeout: Maximum total execution time in seconds

    Returns:
        AgenticLoopResult with chunks, history_delta, final_text, etc.

    Raises:
        RuntimeError: If called from within an async context
        TimeoutError: If execution exceeds timeout
    """
    # Check if we're in an async context
    try:
        asyncio.get_running_loop()
        # If we reach here, there's a running loop (async context)
        raise RuntimeError(
            "execute_with_tools_sync() cannot be called from an async context. "
            "Use 'await execute_with_tools()' instead."
        )
    except RuntimeError as e:
        # Check if this is the expected "no running event loop" error
        error_msg = str(e).lower()
        if "no running event loop" not in error_msg and "no running loop" not in error_msg:
            # Unexpected RuntimeError - re-raise
            raise
        # No running loop - safe to proceed with asyncio.run()

    # Run the async function
    return asyncio.run(
        execute_with_tools(
            provider=provider,
            history=history,
            system_prompt=system_prompt,
            mcp_client=mcp_client,
            max_iterations=max_iterations,
            timeout=timeout,
        )
    )
