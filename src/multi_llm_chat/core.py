import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import google.generativeai as genai
import openai
from dotenv import load_dotenv

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

load_dotenv()


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


def list_gemini_models():
    """List available Gemini models"""
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
        return

    genai.configure(api_key=GOOGLE_API_KEY)
    print("利用可能なGeminiモデル:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  - {m.name}")


def call_gemini_api(history, system_prompt=None):
    """Call Gemini API with optional system prompt

    DEPRECATED (Will be removed in future version):
    This function uses a global shared provider instance which causes
    prompt pollution in concurrent environments (e.g., Gradio sessions).

    **DO NOT USE in production code.**

    Migration path:
    - For session-scoped: ChatService with injected providers
    - For one-off calls: llm_provider.create_provider("gemini")

    SECURITY WARNING: Using this in multi-user environments WILL result in:
    - System prompt leaking between users
    - Cached models being shared across sessions
    - Race conditions in concurrent requests

    This function is kept only for backward compatibility with existing tests.
    """
    import warnings

    warnings.warn(
        "call_gemini_api() is deprecated and will be removed. "
        "Use ChatService or create_provider() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        provider = create_provider("gemini")
        yield from provider.call_api(history, system_prompt)
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


def call_chatgpt_api(history, system_prompt=None):
    """Call ChatGPT API with optional system prompt

    DEPRECATED (Will be removed in future version):
    This function uses a global shared provider instance which causes
    prompt pollution in concurrent environments (e.g., Gradio sessions).

    **DO NOT USE in production code.**

    Migration path:
    - For session-scoped: ChatService with injected providers
    - For one-off calls: llm_provider.create_provider("chatgpt")

    SECURITY WARNING: Using this in multi-user environments WILL result in:
    - System prompt leaking between users
    - Client instances being shared across sessions
    - Race conditions in concurrent requests

    This function is kept only for backward compatibility with existing tests.
    """
    import warnings

    warnings.warn(
        "call_chatgpt_api() is deprecated and will be removed. "
        "Use ChatService or create_provider() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        provider = create_provider("chatgpt")
        yield from provider.call_api(history, system_prompt)
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


def stream_text_events(history, provider_name, system_prompt=None):
    """Stream normalized text events from a provider.

    Args:
        history: Conversation history for the request
        provider_name: Provider identifier ("gemini" or "chatgpt")
        system_prompt: Optional system instruction

    Yields:
        str: Normalized text chunks
    """
    provider = create_provider(provider_name)
    yield from provider.stream_text_events(history, system_prompt)


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
async def execute_with_tools(
    provider: Any,
    history: List[Dict],
    system_prompt: Optional[str] = None,
    mcp_client: Optional[Any] = None,
    max_iterations: int = 10,
    timeout: float = 120.0,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute LLM call with Agentic Loop for tool execution.

    Repeatedly calls LLM and executes tools until:
    - LLM returns text without tool calls
    - max_iterations is reached
    - timeout is exceeded
    - An error occurs

    Args:
        provider: LLM provider instance (Gemini/ChatGPT)
        history: Conversation history (will be mutated)
        system_prompt: Optional system prompt
        mcp_client: Optional MCP client for tool execution
        max_iterations: Maximum number of LLM calls (default: 10)
        timeout: Maximum total execution time in seconds (default: 120)
        tools: Optional tool definitions (JSON Schema format).
               If None, tools will be fetched from mcp_client.

    Yields:
        Streaming chunks from LLM:
        - {"type": "text", "content": str}
        - {"type": "tool_call", "content": {...}}
        - {"type": "tool_result", "content": {...}}

    Raises:
        TimeoutError: If execution exceeds timeout
        ValueError: If tool call is requested but mcp_client is None
    """
    import logging
    import time

    logger = logging.getLogger(__name__)
    start_time = time.time()

    # Use provided tools if any, otherwise list from MCP client
    if tools is None and mcp_client:
        tools = await mcp_client.list_tools()

    _iteration = 0
    for _iteration in range(max_iterations):
        # Check timeout before each iteration
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Agentic loop exceeded {timeout}s timeout")

        # Call LLM
        tool_calls_in_turn = []
        thought_text = ""
        for chunk in provider.call_api(history, system_prompt, tools):
            # Check timeout during streaming
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Agentic loop exceeded {timeout}s timeout")

            chunk_type = chunk.get("type")

            if chunk_type == "text":
                content = chunk.get("content", "")
                thought_text += content
                yield chunk
            elif chunk_type == "tool_call":
                tool_call = chunk.get("content", {})
                tool_calls_in_turn.append(tool_call)
                yield chunk  # Notify UI

        # If no tool calls, loop ends (final response)
        if not tool_calls_in_turn:
            # Add final text to history if it exists
            if thought_text:
                history.append(
                    {"role": provider.name, "content": [{"type": "text", "content": thought_text}]}
                )
            break

        # Tool calls received - execute them
        if tool_calls_in_turn and not mcp_client:
            logger.error("Tool call received but mcp_client is None")
            raise ValueError("MCP client is required for tool execution")

        # Execute tools and collect results
        tool_results = []
        for tool_call in tool_calls_in_turn:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            tool_call_id = tool_call.get("tool_call_id")  # OpenAI only

            try:
                result = await mcp_client.call_tool(name, arguments)

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

                # Notify UI
                yield {"type": "tool_result", "content": tool_result}

            except ConnectionError:
                # MCP server is down - cannot continue
                logger.error("MCP connection error during tool execution: %s", name)
                raise
            except Exception as e:
                # Tool execution failed - report to LLM
                logger.warning("Tool execution failed for %s: %s", name, e)
                error_text = f"Tool execution failed: {str(e)}"
                tool_result = {
                    "name": name,
                    "content": error_text,
                }
                if tool_call_id:
                    tool_result["tool_call_id"] = tool_call_id

                tool_results.append(tool_result)
                yield {"type": "tool_result", "content": tool_result}

        # Append tool_call and tool_result to history
        # Note: History format is structured content (list of items)
        assistant_entry = {"role": provider.name, "content": []}
        if thought_text:
            assistant_entry["content"].append({"type": "text", "content": thought_text})
        for tc in tool_calls_in_turn:
            assistant_entry["content"].append({"type": "tool_call", "content": tc})
        history.append(assistant_entry)

        # Tool results as separate user message
        user_entry = {"role": "user", "content": []}
        for tr in tool_results:
            tool_result_item = {"type": "tool_result", "content": tr["content"]}
            if "tool_call_id" in tr:
                tool_result_item["tool_call_id"] = tr["tool_call_id"]
            if "name" in tr:
                tool_result_item["name"] = tr["name"]
            user_entry["content"].append(tool_result_item)
        history.append(user_entry)
    else:
        # Loop reached max_iterations without breaking
        logger.warning("Agentic loop reached max_iterations=%d", max_iterations)
