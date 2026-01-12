"""LLM Provider abstraction layer using Strategy pattern

This module provides a unified interface for different LLM providers (Gemini, ChatGPT, etc.),
making it easy to add new providers without modifying existing code.
"""

import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import openai
from dotenv import load_dotenv
from google.generativeai.types import FunctionDeclaration, Tool

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from .history_utils import content_to_text
from .token_utils import estimate_tokens, get_buffer_factor, get_max_context_length

load_dotenv()

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo")

# Feature flags
MCP_ENABLED = os.getenv("MULTI_LLM_CHAT_MCP_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)


# MCP Tool conversion functions


def mcp_tools_to_gemini_format(mcp_tools: List[Dict[str, Any]]) -> Optional[List[Tool]]:
    """Convert MCP tool definitions to Gemini Tool format.

    Args:
        mcp_tools: List of MCP tool definitions with structure:
            [{"name": str, "description": str, "inputSchema": dict}, ...]

    Returns:
        A list containing a single Gemini Tool object, or None if no tools are provided.
    """
    if not mcp_tools:
        return None

    function_declarations = []
    for tool in mcp_tools:
        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("inputSchema")

        if not name:
            logger.warning(
                "Skipping MCP tool without name. Tool data: %s",
                {k: v for k, v in tool.items() if k != "inputSchema"},
            )
            continue

        func_decl = FunctionDeclaration(
            name=name,
            description=description,
            parameters=parameters,
        )
        function_declarations.append(func_decl)

    if not function_declarations:
        return None

    return [Tool(function_declarations=function_declarations)]


def mcp_tools_to_openai_format(mcp_tools: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Convert MCP tool definitions to OpenAI tools format.

    Args:
        mcp_tools: List of MCP tool definitions with structure:
            [{"name": str, "description": str, "inputSchema": dict}, ...]

    Returns:
        List of OpenAI tool definitions:
            [{"type": "function", "function": {"name": str, ...}}, ...]
        or None if no tools are provided.
    """
    if not mcp_tools:
        return None

    openai_tools = []
    for tool in mcp_tools:
        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("inputSchema")

        if not name:
            logger.warning(
                "Skipping MCP tool without name. Tool data: %s",
                {k: v for k, v in tool.items() if k != "inputSchema"},
            )
            continue

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or "",
                    "parameters": parameters or {},
                },
            }
        )

    return openai_tools if openai_tools else None


def parse_openai_tool_call(tool_call: dict) -> Dict[str, Any]:
    """Parse OpenAI tool_call to common format.

    Args:
        tool_call: OpenAI tool_call object with structure:
            {"id": str, "type": "function", "function": {"name": str, "arguments": str}}

    Returns:
        Common format dict:
            {"name": str, "arguments": dict, "tool_call_id": str}
    """
    function = tool_call.get("function", {})
    name = function.get("name")
    args_json = function.get("arguments", "{}")
    tool_call_id = tool_call.get("id")

    # Parse JSON arguments
    try:
        arguments = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse tool arguments JSON: %s", e)
        arguments = {}

    return {"name": name, "arguments": arguments, "tool_call_id": tool_call_id}


def _parse_tool_response_payload(response_payload):
    """Parse tool response payload into a dictionary format.

    Args:
        response_payload: Tool response content

    Returns:
        dict: Parsed payload suitable for Gemini function_response

    Raises:
        TypeError: If payload type cannot be safely converted

    Note:
        Handles dict, str, int, float, list, bool, and None.
        Unexpected types trigger a warning and attempt str() conversion.
    """
    if response_payload is None:
        return {}

    if isinstance(response_payload, dict):
        return response_payload

    if isinstance(response_payload, str):
        try:
            parsed = json.loads(response_payload)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except (json.JSONDecodeError, ValueError, RecursionError) as e:
            logger.debug("Failed to parse JSON payload: %s", e)
            return {"result": response_payload}

    # Whitelist of allowed types: dict, str, int, float, list, bool
    if isinstance(response_payload, (int, float, list, bool)):
        return {"result": response_payload}

    # Unexpected type: log warning and attempt safe conversion
    logger.warning(
        "Tool response payload has unexpected type: %s. Attempting str() conversion.",
        type(response_payload).__name__,
    )
    try:
        return {"result": str(response_payload)}
    except (TypeError, AttributeError) as e:
        logger.error("Failed to convert tool response to string: %s", e)
        raise TypeError(
            f"Tool response payload type {type(response_payload).__name__} "
            f"cannot be safely converted to dict"
        ) from e


class OpenAIToolCallAssembler:
    """Assembles OpenAI API streaming tool calls.

    OpenAI API streams tool calls where arguments arrive as partial JSON strings
    across multiple chunks. Each tool call has an 'index' to identify it in
    parallel function calling scenarios.

    State Management:
        _tools_by_index: Dict[int, Dict]
            Key: tool_call.index
            Value: {
                "id": str,
                "name": str,
                "arguments_json": str,  # Accumulated JSON string
                "complete": bool
            }

    Design:
        - Arguments arrive as incremental JSON strings: '{"loc' → 'ation": "T' → 'okyo"}'
        - Parse JSON only when complete (end of stream or finish_reason)
        - Multiple tool calls identified by index attribute

    Usage:
        ```python
        assembler = OpenAIToolCallAssembler()
        for chunk in stream:
            if chunk.choices[0].delta.tool_calls:
                for tc_delta in chunk.choices[0].delta.tool_calls:
                    result = assembler.process_tool_call(tc_delta)
                    if result:
                        yield result
        # Finalize any pending calls at stream end
        for result in assembler.finalize_pending_calls():
            yield result
        ```
    """

    def __init__(self):
        self._tools_by_index: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        """Clear all internal state for reuse."""
        self._tools_by_index.clear()

    def process_tool_call(self, tool_call_delta) -> Optional[Dict[str, Any]]:
        """Process tool_call delta from streaming response.

        Args:
            tool_call_delta: Tool call delta with structure:
                {
                    "index": int,
                    "id": str (optional, first chunk only),
                    "function": {
                        "name": str (optional, first chunk only),
                        "arguments": str (partial JSON fragment)
                    }
                }

        Returns:
            Optional[Dict[str, Any]]: {"type": "tool_call", "content": {...}} if complete,
                                      None if still accumulating
        """
        index = tool_call_delta.index
        function = tool_call_delta.function

        # Initialize tool call entry if first chunk
        if index not in self._tools_by_index:
            self._tools_by_index[index] = {
                "id": getattr(tool_call_delta, "id", None),
                "name": None,
                "arguments_json": "",
                "complete": False,
            }

        tool_call = self._tools_by_index[index]

        # Update id if present (first chunk)
        if hasattr(tool_call_delta, "id") and tool_call_delta.id:
            tool_call["id"] = tool_call_delta.id

        # Update name if present (first chunk)
        if hasattr(function, "name") and function.name:
            tool_call["name"] = function.name

        # Accumulate arguments (incremental JSON string)
        if hasattr(function, "arguments") and function.arguments:
            tool_call["arguments_json"] += function.arguments

        # Note: We don't emit immediately - wait for finalize_pending_calls()
        # to ensure complete JSON before parsing
        return None

    def finalize_pending_calls(self):
        """Finalize all pending tool calls at end of stream.

        Parses accumulated JSON arguments and yields complete tool calls.

        Yields:
            Dict[str, Any]: {"type": "tool_call", "content": {...}}
        """
        for index in sorted(self._tools_by_index.keys()):
            tool_call = self._tools_by_index[index]
            if not tool_call["complete"]:
                # Parse accumulated JSON
                try:
                    args_json = tool_call["arguments_json"]
                    arguments = json.loads(args_json) if args_json else {}
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse tool call arguments for index %s: %s", index, e)
                    arguments = {}

                logger.debug(
                    "Finalizing tool_call: index=%s, id=%s, name=%s",
                    index,
                    tool_call["id"],
                    tool_call["name"],
                )

                yield {
                    "type": "tool_call",
                    "content": {
                        "name": tool_call["name"],
                        "arguments": arguments,
                        "tool_call_id": tool_call["id"],
                    },
                }
                tool_call["complete"] = True


class GeminiToolCallAssembler:
    """Assembles Gemini API streaming tool calls with hybrid tracking.

    Gemini API streams tool calls where the name and arguments can arrive in separate
    chunks. For parallel function calling, each part has an 'index' attribute to
    correctly match names with their arguments.

    State Management:
        _tools_by_index: Dict[int, Dict]
            Used for indexed parts (parallel function calling)
            Key: part.index
            Value: {"name": str, "arguments": dict, "complete": bool}
        _sequential_queue: List[int]
            FIFO queue of sequential (non-indexed) tool call indexes
        _next_sequential_index: int
            Counter for non-indexed parts (starts at -1, decrements)

    Design:
        - Indexed parts (part.index exists): Use dictionary-based tracking
        - Non-indexed parts: Use FIFO queue with negative indexes
        - Emits tool_call event when both name and args are received
        - Stream end → emit all remaining calls (may have empty args)

    Usage:
        Currently, a new instance is created for each API call:
        ```python
        assembler = GeminiToolCallAssembler()
        for chunk in stream:
            result = assembler.process_function_call(part, function_call)
        ```

        For future instance reuse, call reset() between API calls:
        ```python
        assembler = GeminiToolCallAssembler()
        # First call
        for chunk in stream1: ...
        assembler.reset()
        # Second call
        for chunk in stream2: ...
        ```
    """

    def __init__(self):
        self._tools_by_index: Dict[int, Dict[str, Any]] = {}
        self._sequential_queue: List[int] = []
        self._next_sequential_index = -1  # Negative indexes for sequential

    def reset(self) -> None:
        """Clear all internal state for reuse.

        Note:
            Currently, a new GeminiToolCallAssembler instance is created
            for each API call. This method is provided for future optimization
            where instance reuse may be implemented.
        """
        self._tools_by_index.clear()
        self._sequential_queue.clear()
        self._next_sequential_index = -1

    def process_function_call(self, chunk_part, function_call) -> Optional[Dict[str, Any]]:
        """Process function_call with hybrid index/FIFO tracking.

        Args:
            chunk_part: The chunk part (may have 'index' attribute)
            function_call: The function_call object (has 'name' and/or 'args')

        Returns:
            Optional[Dict[str, Any]]: {"type": "tool_call", "content": {...}} if complete,
                                      None if still waiting for more chunks
        """
        # Check if part has explicit index (parallel tool calling)
        explicit_index = getattr(chunk_part, "index", None)

        # Determine which tracking method to use
        if explicit_index is not None:
            # Parallel tool calling: use explicit index
            index = explicit_index
        else:
            # Sequential tool calling: use FIFO strategy
            has_name = hasattr(function_call, "name") and function_call.name
            has_args = hasattr(function_call, "args") and function_call.args is not None

            if has_name and has_args:
                # Both name and args in same chunk: process immediately without queue
                # This prevents queue pollution when fast responses arrive in single chunk
                index = self._next_sequential_index
                self._next_sequential_index -= 1
            elif has_name:
                # Name chunk only: create new sequential entry
                index = self._next_sequential_index
                self._next_sequential_index -= 1
                self._sequential_queue.append(index)
            elif has_args and self._sequential_queue:
                # Args chunk: match with oldest incomplete sequential tool
                index = self._sequential_queue.pop(0)
            else:
                # Orphaned args (no pending tool): create new entry
                index = self._next_sequential_index
                self._next_sequential_index -= 1

        # Initialize tool call entry if not exists
        if index not in self._tools_by_index:
            self._tools_by_index[index] = {"name": None, "arguments": {}, "complete": False}

        tool_call = self._tools_by_index[index]

        # Update name if present
        if hasattr(function_call, "name") and function_call.name:
            tool_call["name"] = function_call.name
            logger.debug("Tool name received: index=%s, name=%s", index, function_call.name)

        # Update arguments if present
        args_updated = False
        if hasattr(function_call, "args") and function_call.args is not None:
            coerced_args = self._coerce_function_args(function_call.args)
            if coerced_args is not None:
                tool_call["arguments"].update(coerced_args)
                args_updated = True
                logger.debug("Tool args received: index=%s, args=%s", index, coerced_args)
            else:
                logger.warning(
                    "Failed to coerce args for tool at index=%s: raw_args=%s",
                    index,
                    function_call.args,
                )

        # Emit if complete (has name and args were just updated)
        if tool_call["name"] and args_updated and not tool_call["complete"]:
            tool_call["complete"] = True
            logger.debug("Emitting tool_call: index=%s, name=%s", index, tool_call["name"])
            return {
                "type": "tool_call",
                "content": {"name": tool_call["name"], "arguments": tool_call["arguments"]},
            }

        return None

    def finalize_pending_calls(self) -> Any:
        """Yield all incomplete tool calls at stream end.

        This handles cases where:
        - Tool call was made without arguments
        - Stream was interrupted before args arrived

        Yields:
            Dict[str, Any]: {"type": "tool_call", "content": {...}}
        """
        for index in sorted(self._tools_by_index.keys()):
            tool_call = self._tools_by_index[index]
            if tool_call["name"] and not tool_call["complete"]:
                logger.debug("Finalizing tool_call: index=%s, name=%s", index, tool_call["name"])
                yield {
                    "type": "tool_call",
                    "content": {"name": tool_call["name"], "arguments": tool_call["arguments"]},
                }

    @staticmethod
    def _coerce_function_args(raw_args: Any) -> Dict[str, Any]:
        """Coerce function arguments to dict format.

        Args:
            raw_args: Raw arguments from Gemini API

        Returns:
            Dict[str, Any]: Coerced dictionary or empty dict if conversion fails
        """
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if hasattr(raw_args, "to_dict"):
            try:
                return raw_args.to_dict()
            except (TypeError, AttributeError) as e:
                logger.warning("Failed to convert to_dict(): %s", e)
                return {}
        try:
            return dict(raw_args)
        except (TypeError, ValueError) as e:
            logger.warning("Failed to convert to dict: %s (type: %s)", e, type(raw_args).__name__)
            return {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def call_api(self, history, system_prompt=None, tools: Optional[List[Dict[str, Any]]] = None):
        """Call the LLM API and return a generator of response chunks

        Args:
            history: List of conversation history dicts with 'role' and 'content'
            system_prompt: Optional system instruction

        Yields:
            Response chunks from the API
        """
        pass

    @abstractmethod
    def extract_text_from_chunk(self, chunk):
        """Extract text content from a response chunk

        Args:
            chunk: Response chunk from the API

        Returns:
            str: Extracted text content
        """
        pass

    @abstractmethod
    def get_token_info(self, text, history=None, model_name=None):
        """Get token usage information

        Args:
            text: Text to analyze
            history: Optional conversation history
            model_name: Optional specific model name (uses default if None)

        Returns:
            dict: Token information with keys 'input_tokens', 'max_tokens'
        """
        pass

    def stream_text_events(self, history, system_prompt=None):
        """Stream normalized text events from the unified dictionary stream."""
        for chunk in self.call_api(history, system_prompt):
            # Support both dict and legacy string format
            if isinstance(chunk, dict):
                if chunk.get("type") == "text":
                    yield chunk.get("content", "")
            elif isinstance(chunk, str):  # Legacy string support
                if chunk:  # Filter empty strings
                    yield chunk


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider with thread-safe LRU caching for models"""

    def __init__(self):
        self._default_model = None
        self._models_cache = OrderedDict()  # LRU cache: hash -> (prompt, model)
        self._cache_max_size = 10  # Limit cache size to prevent memory leak
        self._cache_lock = threading.Lock()  # Protect cache operations
        self._configure()

    def _configure(self):
        """Configure the Gemini SDK if an API key is available"""
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            return True
        return False

    @staticmethod
    def _hash_prompt(prompt):
        """Generate SHA256 hash for a prompt to use as cache key"""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _get_model(self, system_prompt=None):
        """Get or create a cached Gemini model instance with thread-safe LRU eviction

        Args:
            system_prompt: Optional system instruction for the model

        Returns:
            GenerativeModel instance
        """
        # If no system prompt, use the default cached model
        if not system_prompt or not system_prompt.strip():
            with self._cache_lock:
                if self._default_model is None:
                    self._default_model = genai.GenerativeModel(GEMINI_MODEL)
                return self._default_model

        # For system prompts, use LRU cache with hash key
        prompt_hash = self._hash_prompt(system_prompt)

        with self._cache_lock:
            if prompt_hash in self._models_cache:
                # Verify prompt hasn't changed (hash collision check)
                cached_prompt, cached_model = self._models_cache[prompt_hash]
                if cached_prompt == system_prompt:
                    # Move to end (most recently used)
                    self._models_cache.move_to_end(prompt_hash)
                    return cached_model
                # Hash collision detected - explicitly evict old entry
                del self._models_cache[prompt_hash]

            # Create new model and add to cache
            model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
            self._models_cache[prompt_hash] = (system_prompt, model)

            # Evict oldest if cache is full
            if len(self._models_cache) > self._cache_max_size:
                self._models_cache.popitem(last=False)

            return model

    @staticmethod
    def format_history(history):
        """Convert structured history to Gemini API format.

        Handles structured history entries containing text, tool calls, and tool results.
        Filters out responses from other LLMs (e.g., ChatGPT).
        """
        gemini_history = []
        for entry in history:
            role = entry.get("role")
            content_list = entry.get("content")

            # Ensure content is always a list for consistent processing
            if content_list is None:
                content_list = []
            elif not isinstance(content_list, list):
                if isinstance(content_list, str):
                    # Handle legacy string format for backward compatibility
                    content_list = [{"type": "text", "content": content_list}]
                else:
                    # Handle unexpected types (int, bool, etc.) defensively
                    content_type = type(content_list).__name__
                    logger.warning(
                        "Unexpected content type: %s, converting to string. "
                        "Expected list or string. Entry role: %s",
                        content_type,
                        entry.get("role"),
                    )
                    content_list = [{"type": "text", "content": str(content_list)}]

            parts = []
            if role == "user":
                # User messages contain text parts
                for item in content_list:
                    if item.get("type") == "text":
                        parts.append({"text": item.get("content", "")})
                if parts:
                    gemini_history.append({"role": "user", "parts": parts})

            elif role == "gemini":
                # Model messages can contain text and function calls
                for item in content_list:
                    if item.get("type") == "text":
                        parts.append({"text": item.get("content", "")})
                    elif item.get("type") == "tool_call":
                        tool_call = item.get("content", {})
                        parts.append(
                            {
                                "function_call": {
                                    "name": tool_call.get("name"),
                                    "args": tool_call.get("arguments", {}),
                                }
                            }
                        )
                if parts:
                    gemini_history.append({"role": "model", "parts": parts})

            elif role == "tool":
                # Tool messages contain function responses
                for item in content_list:
                    if item.get("type") == "tool_result":
                        response_payload = _parse_tool_response_payload(item.get("content"))
                        tool_call_id = item.get("tool_call_id")
                        function_response = {
                            "name": item.get("name"),
                            "response": response_payload,
                        }
                        if tool_call_id:
                            function_response["id"] = tool_call_id
                        parts.append({"function_response": function_response})
                if parts:
                    gemini_history.append({"role": "function", "parts": parts})

            # Legacy handling for "tool_calls" key (can be removed later)
            if entry.get("tool_calls") and role == "gemini":
                legacy_parts = []
                for tool_call in entry.get("tool_calls"):
                    legacy_parts.append(
                        {
                            "function_call": {
                                "name": tool_call.get("name"),
                                "args": tool_call.get("arguments", {}),
                            }
                        }
                    )
                if legacy_parts:
                    # Merge with existing parts if any
                    if gemini_history and gemini_history[-1]["role"] == "model":
                        gemini_history[-1]["parts"].extend(legacy_parts)
                    else:
                        gemini_history.append({"role": "model", "parts": legacy_parts})

        return gemini_history

    def call_api(
        self,
        history: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Call Gemini API and yield response chunks with safety error handling

        Args:
            history: Conversation history in structured format
            system_prompt: Optional system instruction
            tools: Optional MCP tools in JSON Schema format
            **kwargs: Additional arguments for future extensions

        Yields:
            Dict[str, Any]: Unified dictionary objects:
                - {"type": "text", "content": str}
                - {"type": "tool_call", "content": {"name": str, "arguments": dict}}
        """
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set")

        model = self._get_model(system_prompt)
        gemini_history = self.format_history(history)
        gemini_tools = mcp_tools_to_gemini_format(tools)

        def _safe_chunk_text(chunk_obj):
            """Extract text from chunk, handling ValueError gracefully."""
            if not hasattr(chunk_obj, "text"):
                return ""
            try:
                return chunk_obj.text or ""
            except ValueError:
                return ""

        assembler = GeminiToolCallAssembler()
        stream_completed_successfully = False

        try:
            response = model.generate_content(gemini_history, stream=True, tools=gemini_tools)

            for chunk in response:
                parts = getattr(chunk, "parts", None)
                if parts:
                    for part in parts:
                        if hasattr(part, "function_call") and part.function_call:
                            tool_event = assembler.process_function_call(part, part.function_call)
                            if tool_event:
                                yield tool_event
                            continue

                        try:
                            part_text = part.text
                        except (ValueError, AttributeError):
                            part_text = ""
                        if part_text:
                            yield {"type": "text", "content": part_text}

                    continue

                chunk_text = _safe_chunk_text(chunk)
                if chunk_text:
                    yield {"type": "text", "content": chunk_text}

            # Mark as successful only if we processed all chunks without exception
            stream_completed_successfully = True

        except genai.types.BlockedPromptException as e:
            raise ValueError(f"Prompt was blocked due to safety concerns: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Gemini API call: {e}")
            raise
        finally:
            # Only emit pending tool calls if stream completed successfully
            if stream_completed_successfully:
                yield from assembler.finalize_pending_calls()

    def extract_text_from_chunk(self, chunk: Any):
        """Extract text from a unified response chunk."""
        if isinstance(chunk, dict) and chunk.get("type") == "text":
            return chunk.get("content", "")
        return ""

    def get_token_info(self, text, history=None, model_name=None):
        """Get token information for Gemini

        Uses estimation with buffer factor (auto-detected or from environment).
        """
        # Use provided model name or fall back to default
        effective_model = model_name if model_name else GEMINI_MODEL

        # Check if history contains tools for buffer factor selection
        has_tools = False
        if history:
            from .history_utils import history_contains_tools

            has_tools = history_contains_tools(history)

        # Get buffer factor (auto-detected based on tool usage)
        buffer_factor = get_buffer_factor(has_tools=has_tools)

        # Calculate tokens for system prompt/text
        text_content = content_to_text(text, include_tool_data=True)
        token_count = int(estimate_tokens(text_content) * buffer_factor)

        # Add history tokens if provided (only count user and gemini messages)
        if history:
            for entry in history:
                role = entry.get("role", "")
                if role in {"user", "gemini", "tool"}:
                    content = content_to_text(entry.get("content", ""), include_tool_data=True)
                    token_count += int(estimate_tokens(content) * buffer_factor)

        # Get max context length for this model
        max_context = get_max_context_length(effective_model)

        return {
            "input_tokens": token_count,
            "max_tokens": max_context,
        }


class ChatGPTProvider(LLMProvider):
    """OpenAI ChatGPT LLM provider (thread-safe)

    The OpenAI client is thread-safe, so concurrent requests can safely share
    the same client instance without additional locking.
    """

    def __init__(self):
        self._client = None
        if OPENAI_API_KEY:
            self._client = openai.OpenAI(api_key=OPENAI_API_KEY)

    @staticmethod
    def format_history(history):
        """Convert history to ChatGPT API format

        Filters out responses from other LLMs (e.g., Gemini) to avoid
        sending ChatGPT messages it didn't generate, which would create
        a self-contradictory conversation.

        Supports structured content including tool_call and tool_result.
        """
        chatgpt_history = []
        for entry in history:
            role = entry["role"]
            content = entry.get("content")

            if role == "system":
                chatgpt_history.append({"role": "system", "content": content_to_text(content)})
            elif role == "user":
                chatgpt_history.append({"role": "user", "content": content_to_text(content)})
            elif role == "chatgpt":
                # Step 1: Normalize content to list
                items = []
                if isinstance(content, list):
                    items = content
                elif isinstance(content, dict):
                    # Single dict is also structured data
                    items = [content]
                else:
                    # Legacy format (string)
                    chatgpt_history.append(
                        {"role": "assistant", "content": content_to_text(content)}
                    )
                    continue

                # Step 2: Group items by type
                text_items = []
                tool_call_items = []
                tool_result_items = []

                for item in items:
                    if not isinstance(item, dict):
                        # String item is treated as text
                        text_items.append(str(item))
                        continue

                    item_type = item.get("type")
                    if item_type == "text":
                        text_items.append(item.get("content", ""))
                    elif item_type == "tool_call":
                        tool_call_items.append(item)
                    elif item_type == "tool_result":
                        tool_result_items.append(item)
                    else:
                        # Unknown type: fallback to content_to_text()
                        text_items.append(content_to_text(item))

                # Step 3: Generate messages (OpenAI allows text and tool_calls in same message)

                # 3-1. Build assistant message if text or tool_calls exist
                if text_items or tool_call_items:
                    message = {"role": "assistant"}

                    # Add text content if exists
                    if text_items:
                        message["content"] = " ".join(text_items)

                    # Add tool_calls if exist
                    if tool_call_items:
                        valid_tool_calls = []
                        for item in tool_call_items:
                            if not item.get("tool_call_id") or not item.get("name"):
                                logger.warning("Skipping incomplete tool_call in history: %s", item)
                                continue
                            valid_tool_calls.append(
                                {
                                    "id": item["tool_call_id"],
                                    "type": "function",
                                    "function": {
                                        "name": item["name"],
                                        "arguments": json.dumps(item["arguments"]),
                                    },
                                }
                            )

                        if valid_tool_calls:
                            message["tool_calls"] = valid_tool_calls

                    # Only append if we have actual content (text) or valid tool_calls
                    # If neither exist, skip this message entirely
                    has_content = "content" in message
                    has_tool_calls = "tool_calls" in message

                    if has_content or has_tool_calls:
                        # OpenAI API specification:
                        # - content and tool_calls can coexist (mixed content is valid)
                        # - content should be None only when no text is present
                        # Reference: https://platform.openai.com/docs/guides/function-calling
                        if has_tool_calls and not text_items:
                            message["content"] = None
                        chatgpt_history.append(message)

                # 3-2. Tool result messages (if any)
                for item in tool_result_items:
                    if not item.get("tool_call_id"):
                        logger.warning("Skipping tool_result without tool_call_id: %s", item)
                        continue
                    chatgpt_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": json.dumps(item.get("result", "")),
                        }
                    )
            # Skip gemini and other roles - they shouldn't be sent to ChatGPT
        return chatgpt_history

    def call_api(self, history, system_prompt=None, tools: Optional[List[Dict[str, Any]]] = None):
        """Call ChatGPT API and yield unified dictionary objects.

        Args:
            history: Conversation history
            system_prompt: Optional system instruction
            tools: Optional MCP tools to convert to OpenAI format

        Yields:
            Dict with "type" and "content" keys:
                {"type": "text", "content": str} or
                {"type": "tool_call", "content": {...}}
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")

        # Build messages for OpenAI format
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        # Filter history to only include user and ChatGPT messages
        chatgpt_history = self.format_history(history)
        messages.extend(chatgpt_history)

        # Prepare API call parameters
        api_params = {"model": CHATGPT_MODEL, "messages": messages, "stream": True}

        # Add tools if provided
        if tools:
            openai_tools = mcp_tools_to_openai_format(tools)
            if openai_tools:
                api_params["tools"] = openai_tools
                api_params["tool_choice"] = "auto"

        # Call API with streaming
        stream = self._client.chat.completions.create(**api_params)

        # Process stream with tool call assembler
        assembler = OpenAIToolCallAssembler()
        for chunk in stream:
            finish_reason = None
            # Check for tool calls and finish reason
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = getattr(choice, "finish_reason", None)

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        assembler.process_tool_call(tool_call_delta)

                # Immediate finalization on tool_calls finish
                if finish_reason == "tool_calls":
                    for result in assembler.finalize_pending_calls():
                        yield result
                    continue  # Skip text processing for tool_calls finish

            # Check for text content
            text_content = self.extract_text_from_chunk(chunk)
            if text_content:
                yield {"type": "text", "content": text_content}

        # Fallback: finalize any remaining calls (defensive)
        for result in assembler.finalize_pending_calls():
            yield result

    def extract_text_from_chunk(self, chunk):
        """Extract text from ChatGPT response chunk

        Handles both string and list responses from OpenAI API.
        """
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                delta_content = delta.get("content")
                if isinstance(delta_content, list):
                    return "".join(str(part) for part in delta_content)
                if delta_content is not None:
                    return delta_content

        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            delta_content = getattr(delta, "content", None)

            # Handle both string and list responses from OpenAI API
            if isinstance(delta_content, list):
                return "".join(
                    part.text if hasattr(part, "text") else str(part) for part in delta_content
                )
            elif delta_content is not None:
                return delta_content
        return ""

    def get_token_info(self, text, history=None, model_name=None, has_tools=False):
        """Get token information for ChatGPT

        Uses tiktoken for accurate counting if available, falls back to estimation.

        Args:
            text: Text content to count tokens for
            history: Optional conversation history
            model_name: Optional model name (uses CHATGPT_MODEL if not provided)
            has_tools: Whether tools are being used (applies buffer factor)
        """
        # Use provided model name or fall back to default
        effective_model = model_name if model_name else CHATGPT_MODEL
        token_count = 0
        use_estimation = not TIKTOKEN_AVAILABLE

        # Try tiktoken for accurate counting if available
        if TIKTOKEN_AVAILABLE:
            try:
                # Map model name to tiktoken encoding
                model_lower = effective_model.lower()
                if "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower:
                    encoding = tiktoken.get_encoding("o200k_base")
                elif "gpt-4" in model_lower:
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:  # gpt-3.5 and others
                    encoding = tiktoken.get_encoding("cl100k_base")

                # Count system prompt/text tokens
                text_content = content_to_text(text, include_tool_data=True)
                token_count = len(encoding.encode(text_content))

                # Add message overhead (3 tokens per message for OpenAI spec)
                token_count += 3

                # Add history tokens if provided (only count user and chatgpt messages)
                if history:
                    for entry in history:
                        role = entry.get("role", "")
                        if role in {"user", "chatgpt"}:
                            content = content_to_text(
                                entry.get("content", ""), include_tool_data=True
                            )
                            token_count += len(encoding.encode(content)) + 3

            except Exception:
                # Fall back to estimation if tiktoken fails
                use_estimation = True

        # Apply buffer factor for tools if needed (even with tiktoken)
        if has_tools:
            buffer_factor = get_buffer_factor(has_tools=True)
            token_count = int(token_count * buffer_factor)

        # Use estimation with buffer (if tiktoken unavailable or failed)
        if use_estimation:
            buffer_factor = get_buffer_factor(has_tools=has_tools)
            text_content = content_to_text(text, include_tool_data=True)
            token_count = int(estimate_tokens(text_content) * buffer_factor)
            if history:
                for entry in history:
                    if entry.get("role") in {"user", "chatgpt"}:
                        token_count += int(
                            estimate_tokens(
                                content_to_text(entry.get("content", ""), include_tool_data=True)
                            )
                            * buffer_factor
                        )

        # Get max context length for this model
        max_context = get_max_context_length(effective_model)

        return {
            "input_tokens": token_count,
            "max_tokens": max_context,
        }


# Provider registry
_PROVIDERS = {"gemini": GeminiProvider, "chatgpt": ChatGPTProvider}

# Cache provider instances for reuse (DEPRECATED: Use create_provider instead)
_PROVIDER_INSTANCES = {}
_provider_lock = threading.Lock()


def create_provider(provider_name):
    """Factory function to create a new provider instance

    Creates a fresh provider instance for session-scoped usage.
    Each call returns a new instance with isolated state (cache, clients).

    Args:
        provider_name: Name of the provider ('gemini', 'chatgpt', etc.)

    Returns:
        LLMProvider: New instance of the requested provider

    Raises:
        ValueError: If provider_name is not registered
    """
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    provider_class = _PROVIDERS[provider_name]
    return provider_class()


def get_provider(provider_name):
    """Factory function to get a provider instance (thread-safe)

    DEPRECATED: This function returns a global shared instance.
    New code should use create_provider() for session-scoped providers.

    Returns cached instance if available to reuse API clients and models.
    Thread-safe for concurrent access in WebUI environment.

    Args:
        provider_name: Name of the provider ('gemini', 'chatgpt', etc.)

    Returns:
        LLMProvider: Instance of the requested provider

    Raises:
        ValueError: If provider_name is not registered
    """
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    # Thread-safe check-and-create pattern
    with _provider_lock:
        if provider_name not in _PROVIDER_INSTANCES:
            provider_class = _PROVIDERS[provider_name]
            _PROVIDER_INSTANCES[provider_name] = provider_class()

    return _PROVIDER_INSTANCES[provider_name]
