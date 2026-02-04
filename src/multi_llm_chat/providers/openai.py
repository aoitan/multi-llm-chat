"""OpenAI ChatGPT provider implementation

Extracted from llm_provider.py as part of Issue #101 refactoring.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import openai

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from ..config import get_config
from ..history_utils import content_to_text
from ..token_utils import estimate_tokens, get_buffer_factor, get_max_context_length
from .base import LLMProvider

logger = logging.getLogger(__name__)


# MCP Tool conversion functions


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

    # OpenAI spec requires 'id' field; warn if missing
    if not tool_call_id:
        logger.warning(
            "OpenAI tool_call missing required 'id' field (name=%s). "
            "This may indicate an invalid API response.",
            name,
        )

    # Parse JSON arguments
    try:
        arguments = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse tool arguments JSON: %s", e)
        arguments = {}

    return {"name": name, "arguments": arguments, "tool_call_id": tool_call_id}


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


class ChatGPTProvider(LLMProvider):
    """OpenAI ChatGPT LLM provider (thread-safe)

    The OpenAI client is thread-safe, so concurrent requests can safely share
    the same client instance without additional locking.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.name = "chatgpt"  # Provider identifier for history tracking
        # Use provided value or fall back to configuration
        config = get_config()
        self.api_key = api_key or config.openai_api_key
        self._client = None
        if self.api_key:
            self._client = openai.OpenAI(api_key=self.api_key)

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
                            if not item.get("tool_call_id"):
                                logger.warning(
                                    "Skipping tool_call without required tool_call_id (name=%s). "
                                    "This may indicate invalid history data.",
                                    item.get("name"),
                                )
                                continue
                            if not item.get("name"):
                                logger.warning("Skipping tool_call without name: %s", item)
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
                        logger.warning(
                            "Skipping tool_result without required tool_call_id (name=%s). "
                            "OpenAI spec requires tool_call_id for tool role messages.",
                            item.get("name"),
                        )
                        continue
                    # OpenAI expects tool content as plain string, not double-JSON-encoded
                    # MCP call_tool() returns {"content": [{"type": "text", "text": "..."}], ...}
                    # which is normalized to {"content": "...", ...} by agentic_loop
                    content = item.get("content", item.get("result", ""))
                    if isinstance(content, dict):
                        # Fallback: if still dict, JSON encode it
                        content = json.dumps(content)
                    chatgpt_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": content,
                        }
                    )
            elif role == "tool":
                # Handle tool results from Agentic Loop
                # role="tool" entries contain tool_result items
                items = content if isinstance(content, list) else [content]
                for item in items:
                    if not isinstance(item, dict):
                        logger.warning("Skipping non-dict item in role='tool' content: %s", item)
                        continue
                    if item.get("type") != "tool_result":
                        logger.warning(
                            "Skipping non-tool_result item in role='tool': type=%s",
                            item.get("type"),
                        )
                        continue
                    if not item.get("tool_call_id"):
                        logger.warning(
                            "Skipping tool_result without required tool_call_id (name=%s)",
                            item.get("name"),
                        )
                        continue
                    chatgpt_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": item.get("content", ""),
                        }
                    )
            # Skip gemini and other roles - they shouldn't be sent to ChatGPT
        return chatgpt_history

    async def call_api(
        self, history, system_prompt=None, tools: Optional[List[Dict[str, Any]]] = None
    ):
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
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        # Build messages for OpenAI format
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        # Filter history to only include user and ChatGPT messages
        chatgpt_history = self.format_history(history)
        messages.extend(chatgpt_history)

        # Get model name from configuration
        config = get_config()
        model = config.chatgpt_model

        # Prepare API call parameters
        api_params = {"model": model, "messages": messages, "stream": True}

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
            model_name: Optional model name (uses config.chatgpt_model if not provided)
            has_tools: Whether tools are being used (applies buffer factor)
        """
        # Use provided model name or fall back to configuration
        if not model_name:
            config = get_config()
            model_name = config.chatgpt_model
        effective_model = model_name
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
