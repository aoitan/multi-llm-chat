import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import openai
from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that depend on them
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# Import after load_dotenv() to ensure env vars are available  # noqa: E402
from .compression import (
    get_pruning_info as _get_pruning_info,
)
from .compression import (
    prune_history_sliding_window as _prune_history_sliding_window,
)
from .history_utils import (
    LLM_ROLES as LLM_ROLES,
)
from .history_utils import (
    get_provider_name_from_model as _get_provider_name_from_model,
)
from .history_utils import (
    prepare_request as _prepare_request,
)
from .history_utils import (
    validate_history_entry as _validate_history_entry,
)
from .llm_provider import (
    CHATGPT_MODEL as CHATGPT_MODEL,
)
from .llm_provider import (
    GEMINI_MODEL as GEMINI_MODEL,
)
from .llm_provider import (
    GOOGLE_API_KEY as GOOGLE_API_KEY,
)
from .llm_provider import (
    MCP_ENABLED as MCP_ENABLED,
)
from .llm_provider import (
    OPENAI_API_KEY as OPENAI_API_KEY,
)
from .llm_provider import (
    ChatGPTProvider as ChatGPTProvider,
)
from .llm_provider import (
    GeminiProvider as GeminiProvider,
)
from .llm_provider import (
    create_provider as create_provider,
)
from .llm_provider import (
    get_provider as get_provider,
)
from .token_utils import (
    estimate_tokens as _estimate_tokens_impl,
)
from .token_utils import (
    get_max_context_length as _get_max_context_length,
)
from .validation import (
    validate_context_length as _validate_context_length,
)
from .validation import (
    validate_system_prompt_length as _validate_system_prompt_length,
)


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


def load_api_key(env_var_name):
    """Load API key from environment"""
    return os.getenv(env_var_name)


def format_history_for_gemini(history):
    """Convert history to Gemini API format

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use GeminiProvider.format_history() directly.
    """
    return GeminiProvider.format_history(history)


def format_history_for_chatgpt(history):
    """Convert history to ChatGPT API format

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use ChatGPTProvider.format_history() directly.
    """
    return ChatGPTProvider.format_history(history)


def list_gemini_models(verbose: bool = True):
    """List available Gemini models (debug utility)

    Args:
        verbose: If True, print models to stdout (default: True)

    Returns:
        list: List of available model names, or empty list if API key not configured
    """
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment variables or .env file.")
        return []

    genai.configure(api_key=GOOGLE_API_KEY)
    models = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            models.append(m.name)
            logger.debug("Available Gemini model: %s", m.name)

    if verbose:
        print("利用可能なGeminiモデル:")
        for name in models:
            print(f"  - {name}")

    logger.info("Found %d Gemini models", len(models))
    return models


async def call_gemini_api_async(history, system_prompt=None):
    """Call Gemini API asynchronously."""
    try:
        provider = create_provider("gemini")
        async for chunk in provider.call_api(history, system_prompt):
            yield chunk
    except ValueError as e:
        yield f"Gemini API Error: {e}"
    except Exception as e:
        error_msg = f"Gemini API Error: An unexpected error occurred: {e}"
        print(error_msg)
        try:
            list_gemini_models()
        except Exception:
            pass
        yield error_msg


def call_gemini_api(history, system_prompt=None):
    """Call Gemini API (synchronous wrapper for backward compatibility)"""
    import asyncio
    import warnings

    warnings.warn(
        "call_gemini_api() is deprecated and will be removed. "
        "Use ChatService or create_provider() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_gen = call_gemini_api_async(history, system_prompt)
    while True:
        try:
            yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            break


async def call_chatgpt_api_async(history, system_prompt=None):
    """Call ChatGPT API asynchronously."""
    try:
        provider = create_provider("chatgpt")
        async for chunk in provider.call_api(history, system_prompt):
            yield chunk
    except ValueError as e:
        yield f"ChatGPT API Error: {e}"
    except openai.APIError as e:
        yield f"ChatGPT API Error: OpenAI APIからエラーが返されました: {e}"
    except openai.APITimeoutError as e:
        yield f"ChatGPT API Error: リクエストがタイムアウトしました: {e}"
    except openai.APIConnectionError as e:
        yield f"ChatGPT API Error: APIへの接続に失敗しました: {e}"
    except Exception as e:
        yield f"ChatGPT API Error: 予期せぬエラーが発生しました: {e}"


def call_chatgpt_api(history, system_prompt=None):
    """Call ChatGPT API (synchronous wrapper for backward compatibility)"""
    import asyncio
    import warnings

    warnings.warn(
        "call_chatgpt_api() is deprecated and will be removed. "
        "Use ChatService or create_provider() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_gen = call_chatgpt_api_async(history, system_prompt)
    while True:
        try:
            yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            break


async def stream_text_events_async(history, provider_name, system_prompt=None):
    """Stream normalized text events from a provider asynchronously."""
    provider = create_provider(provider_name)
    async for chunk in provider.stream_text_events(history, system_prompt):
        yield chunk


def stream_text_events(history, provider_name, system_prompt=None):
    """Stream normalized text events (synchronous wrapper for backward compatibility)"""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_gen = stream_text_events_async(history, provider_name, system_prompt)
    while True:
        try:
            yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            break


def extract_text_from_chunk(chunk, model_name):
    """Extract text content from API response chunk

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.extract_text_from_chunk() directly.
    """
    try:
        provider_name = _get_provider_name_from_model(model_name)
        provider = create_provider(provider_name)
        return provider.extract_text_from_chunk(chunk)
    except Exception:
        if isinstance(chunk, str):
            return chunk
        return ""


def prepare_request(history, system_prompt, model_name):
    """Prepare API request with system prompt and history"""
    return _prepare_request(history, system_prompt, model_name)


def _estimate_tokens(text):
    return _estimate_tokens_impl(text)


def get_max_context_length(model_name):
    return _get_max_context_length(model_name)


def calculate_tokens(text: str, model_name: str) -> int:
    """Calculate token count for text using model-appropriate method"""
    provider_name = _get_provider_name_from_model(model_name)
    provider = get_provider(provider_name)

    result = provider.get_token_info(text, history=None, model_name=model_name)

    return result["input_tokens"]


def get_token_info(
    text: str, model_name: str, history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Get token information for the given text and model

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.get_token_info() directly.
    """
    from .llm_provider import TIKTOKEN_AVAILABLE

    # Determine provider from model name
    provider_name = _get_provider_name_from_model(model_name)

    # Get provider and delegate token calculation with actual model name
    provider = get_provider(provider_name)

    result = provider.get_token_info(text, history, model_name=model_name)

    # Add is_estimated flag for backward compatibility
    is_estimated = provider_name == "gemini" or not TIKTOKEN_AVAILABLE

    return {
        "token_count": result["input_tokens"],
        "max_context_length": result["max_tokens"],
        "is_estimated": is_estimated,
    }


def prune_history_sliding_window(history, max_tokens, model_name, system_prompt=None):
    """Prune conversation history using sliding window approach"""

    return _prune_history_sliding_window(
        history, max_tokens, model_name, system_prompt, token_calculator=calculate_tokens
    )


def get_pruning_info(history, max_tokens, model_name, system_prompt=None):
    """Get information about how history would be pruned"""

    return _get_pruning_info(
        history, max_tokens, model_name, system_prompt, token_calculator=calculate_tokens
    )


def validate_system_prompt_length(system_prompt, model_name):
    """Validate that system prompt doesn't exceed model's max context"""

    return _validate_system_prompt_length(
        system_prompt, model_name, token_calculator=calculate_tokens
    )


def validate_context_length(history, system_prompt, model_name):
    """Validate that system prompt + latest turn doesn't exceed max context"""

    return _validate_context_length(
        history, system_prompt, model_name, token_calculator=calculate_tokens
    )


# Agentic Loop implementation
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
    import copy
    import logging
    import time

    logger = logging.getLogger(__name__)
    start_time = time.time()

    # Create working copy of history (do not mutate original)
    working_history = copy.deepcopy(history)
    original_history_length = len(history)
    chunks = []
    final_text = ""
    timed_out = False
    error = None

    # Use provided tools if any, otherwise list from MCP client
    if tools is None and mcp_client:
        tools = await mcp_client.list_tools()

    iterations_count = 0
    try:
        for _iteration_index in range(max_iterations):
            # Check timeout before each iteration
            if time.time() - start_time > timeout:
                logger.warning("Agentic loop timed out after %.1f seconds", timeout)
                raise TimeoutError(f"Execution exceeded timeout of {timeout} seconds")

            # Call LLM
            tool_calls_in_turn = []
            thought_text = ""
            async for chunk in provider.call_api(working_history, system_prompt, tools):
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
                            "tool_call_id": tc.get("tool_call_id") or tc.get("name") or "unknown",
                        }
                        for tc in tool_calls_in_turn
                    ],
                }
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

                    # Extract text content (simplify for LLM)
                    text_parts = [
                        item.get("text", "")
                        for item in result.get("content", [])
                        if item.get("type") == "text"
                    ]
                    result_text = "\n".join(text_parts) if text_parts else "(no text output)"

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
                tool_result_item = {
                    "type": "tool_result",
                    "name": tr.get("name"),
                    "content": tr["content"],
                    # tool_call_id is required; fallback to name if missing (shouldn't happen)
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
        raise
    except Exception as e:
        error = str(e)
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
