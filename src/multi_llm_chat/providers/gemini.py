"""Google Gemini provider implementation

Extracted from llm_provider.py as part of Issue #101 refactoring.
Updated for Adapter pattern as part of Issue #136.
Updated for SDK switching as part of Issue #137.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from google.generativeai.types import FunctionDeclaration, Tool

from ..config import get_config
from ..history_utils import content_to_text
from ..token_utils import estimate_tokens, get_buffer_factor, get_max_context_length
from .base import LLMProvider
from .gemini_adapter import LegacyGeminiAdapter
from .gemini_client import NewGeminiAdapter

logger = logging.getLogger(__name__)

# MCP Tool conversion functions (Gemini)


def _sanitize_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove JSON Schema fields that Gemini API doesn't accept.

    Gemini only accepts: type, properties, required, description, items, enum.
    Removes validation keywords like minItems, maxItems, pattern, format, etc.

    Args:
        schema: JSON Schema dictionary

    Returns:
        Sanitized schema with only Gemini-compatible fields
    """
    if not isinstance(schema, dict):
        return schema

    # Fields that Gemini accepts
    allowed_fields = {
        "type",
        "properties",
        "required",
        "description",
        "items",
        "enum",
    }

    # Recursively clean the schema
    cleaned = {}
    for key, value in schema.items():
        if key not in allowed_fields:
            continue

        if key == "properties" and isinstance(value, dict):
            # Recursively clean nested properties
            cleaned[key] = {
                prop_name: _sanitize_schema_for_gemini(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            # Recursively clean array items schema
            cleaned[key] = _sanitize_schema_for_gemini(value)
        else:
            cleaned[key] = value

    return cleaned


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

        # Sanitize schema to remove fields Gemini doesn't accept
        if parameters and isinstance(parameters, dict):
            parameters = _sanitize_schema_for_gemini(parameters)

        func_decl = FunctionDeclaration(
            name=name,
            description=description,
            parameters=parameters,
        )
        function_declarations.append(func_decl)

    if not function_declarations:
        return None

    return [Tool(function_declarations=function_declarations)]


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


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider with thread-safe LRU caching for models"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.name = "gemini"  # Provider identifier for history tracking
        # Use provided values or fall back to configuration
        config = get_config()
        self.api_key = api_key or config.google_api_key
        self.model_name = model_name or config.gemini_model

        # SDK switching via environment variable (Issue #137)
        use_new_sdk = os.getenv("USE_NEW_GEMINI_SDK", "0") == "1"
        adapter_class = NewGeminiAdapter if use_new_sdk else LegacyGeminiAdapter

        # Use adapter pattern for SDK abstraction (Issue #136)
        self._adapter = adapter_class(self.api_key) if self.api_key else None

    def _get_model(self, system_prompt=None):
        """Get or create a cached Gemini model instance

        Args:
            system_prompt: Optional system instruction for the model

        Returns:
            GenerativeModel instance (via adapter)
        """
        if not self._adapter:
            # In test environments, adapter might not be initialized
            # Try to create it with the current api_key (which might be None for mocked tests)
            use_new_sdk = os.getenv("USE_NEW_GEMINI_SDK", "0") == "1"
            adapter_class = NewGeminiAdapter if use_new_sdk else LegacyGeminiAdapter
            self._adapter = adapter_class(self.api_key)
        return self._adapter.get_model(self.model_name, system_prompt)

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
                        # Support both nested ({content: {...}}) and flattened formats
                        tool_call_content = item.get("content")
                        if isinstance(tool_call_content, dict) and tool_call_content:
                            name = tool_call_content.get("name")
                            args = tool_call_content.get("arguments", {})
                        else:
                            # Flattened: {type: "tool_call", name, arguments, ...}
                            name = item.get("name")
                            args = item.get("arguments", {})
                        parts.append(
                            {
                                "function_call": {
                                    "name": name,
                                    "args": args,
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

    async def call_api(
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
        if not self.api_key:
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
            with self._adapter.handle_api_errors():
                response = model.generate_content(gemini_history, stream=True, tools=gemini_tools)

                for chunk in response:
                    parts = getattr(chunk, "parts", None)
                    if parts:
                        for part in parts:
                            if hasattr(part, "function_call") and part.function_call:
                                tool_event = assembler.process_function_call(
                                    part, part.function_call
                                )
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

        except ValueError:
            # Re-raise ValueError (including converted BlockedPromptException)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Gemini API call: {e}")
            raise
        finally:
            # Only emit pending tool calls if stream completed successfully
            if stream_completed_successfully:
                for result in assembler.finalize_pending_calls():
                    yield result

    def extract_text_from_chunk(self, chunk: Any):
        """Extract text from a unified response chunk."""
        if isinstance(chunk, dict) and chunk.get("type") == "text":
            return chunk.get("content", "")
        return ""

    def get_token_info(self, text, history=None, model_name=None, has_tools=False):
        """Get token information for Gemini

        Uses estimation with buffer factor (auto-detected or from environment).
        """
        # Use provided model name or fall back to default
        effective_model = model_name if model_name else self.model_name

        # Check if history contains tools for buffer factor selection
        if not has_tools and history:
            from ..history_utils import history_contains_tools

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
