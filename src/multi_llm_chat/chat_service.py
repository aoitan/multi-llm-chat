"""ChatService - business logic layer for chat operations

This module encapsulates core chat logic (mention parsing, LLM routing, history management)
independent of UI implementation. This allows both CLI and WebUI to share
the same business logic without duplication.
"""

import logging

from .core import AgenticLoopResult, execute_with_tools_stream
from .llm_provider import create_provider
from .mcp import get_mcp_manager

logger = logging.getLogger(__name__)


def parse_mention(message):
    """Parse mention from user message

    Args:
        message: User message string

    Returns:
        str or None: "gemini", "chatgpt", "all", or None if no mention
    """
    msg_stripped = message.strip()
    if msg_stripped.startswith("@gemini"):
        return "gemini"
    elif msg_stripped.startswith("@chatgpt"):
        return "chatgpt"
    elif msg_stripped.startswith("@all"):
        return "all"
    return None


ASSISTANT_LABELS = {
    "assistant": "**Assistant:**\n",
    "gemini": "**Gemini:**\n",
    "chatgpt": "**ChatGPT:**\n",
}

# Display name mapping for consistent label formatting
PROVIDER_DISPLAY_NAMES = {
    "gemini": "Gemini",
    "chatgpt": "ChatGPT",
}


class ChatService:
    """Business logic layer for chat operations

    Encapsulates core chat logic (mention parsing, LLM routing, history management)
    independent of UI implementation. This allows both CLI and WebUI to share
    the same business logic without duplication.

    Attributes:
        display_history: UI-friendly history format [[user_msg, assistant_msg], ...]
        logic_history: API-friendly history format [{"role": "user", "content": "..."}]
        system_prompt: System prompt text for LLM context
        gemini_provider: Gemini LLM provider instance
        chatgpt_provider: ChatGPT LLM provider instance
        mcp_client: Optional MCP client for tool execution
    """

    def __init__(
        self,
        display_history=None,
        logic_history=None,
        system_prompt="",
        gemini_provider=None,
        chatgpt_provider=None,
        mcp_client=None,
    ):
        """Initialize ChatService with optional existing state and providers

        Args:
            display_history: Optional existing display history
            logic_history: Optional existing logic history
            system_prompt: Optional system prompt (default: "")
            gemini_provider: Optional Gemini provider instance (lazy-created if None)
            chatgpt_provider: Optional ChatGPT provider instance (lazy-created if None)
            mcp_client: Optional MCP client for tool execution.
                       If None, will use global MCPServerManager from get_mcp_manager()
        """
        self.display_history = display_history if display_history is not None else []
        self.logic_history = logic_history if logic_history is not None else []
        self.system_prompt = system_prompt

        # Store injected providers or None for lazy initialization
        self._gemini_provider = gemini_provider
        self._chatgpt_provider = chatgpt_provider

        # Use provided mcp_client or fall back to global MCPServerManager
        if mcp_client is not None:
            self.mcp_client = mcp_client
        else:
            # Try to get global MCPServerManager
            self.mcp_client = get_mcp_manager()

    @property
    def gemini_provider(self):
        """Lazy-initialized Gemini provider"""
        if self._gemini_provider is None:
            self._gemini_provider = create_provider("gemini")
        return self._gemini_provider

    @property
    def chatgpt_provider(self):
        """Lazy-initialized ChatGPT provider"""
        if self._chatgpt_provider is None:
            self._chatgpt_provider = create_provider("chatgpt")
        return self._chatgpt_provider

    def _handle_api_error(self, error, provider_name):
        """Handle API errors in a consistent way

        Args:
            error: Exception that occurred
            provider_name: Name of the provider ("gemini" or "chatgpt")
        """
        provider_title = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.capitalize())
        label = ASSISTANT_LABELS.get(provider_name, f"**{provider_title}:**\n")

        if isinstance(error, ValueError):
            # API key errors
            error_msg = f"[System: エラー - {str(error)}]"
        else:
            # Other API errors (network, blocked prompts, etc.)
            error_msg = f"[System: {provider_title} APIエラー - {str(error)}]"

        self.display_history[-1]["content"] = f"{label}{error_msg}"
        self.logic_history.append(
            {
                "role": provider_name,
                "content": [{"type": "text", "content": error_msg}],
            }
        )

    async def process_message(self, user_message, tools=None):
        """Process user message and generate LLM responses

        This is an async generator function that yields intermediate states
        for streaming UI updates.

        Args:
            user_message: User's input message
            tools: Optional list of tools for the LLM

        tuple:
            tuple: (display_history, logic_history, chunk) after each update
        """
        mention = parse_mention(user_message)

        # Add user message to histories (structured format)
        user_entry = {"role": "user", "content": [{"type": "text", "content": user_message}]}
        self.logic_history.append(user_entry)
        self.display_history.append({"role": "user", "content": user_message})
        yield self.display_history, self.logic_history, {"type": "text", "content": ""}

        # If no mention, treat as memo (no LLM call)
        if mention is None:
            return

        # For @all, create lightweight snapshot; execute_with_tools_stream will deep-copy internally
        history_at_start = list(self.logic_history) if mention == "all" else None

        # Process models
        models_to_call = []
        if mention == "all":
            models_to_call = ["gemini", "chatgpt"]
        else:
            models_to_call = [mention]

        for model_name in models_to_call:
            provider = self.gemini_provider if model_name == "gemini" else self.chatgpt_provider
            label = ASSISTANT_LABELS[model_name]

            # Add assistant entry for this model's response
            self.display_history.append({"role": "assistant", "content": label})

            yield self.display_history, self.logic_history, {"type": "text", "content": ""}

            # Prepare input history for this model
            # execute_with_tools_stream will deep-copy it internally
            if mention == "all":
                input_history = history_at_start
            else:
                input_history = self.logic_history

            any_yielded = False
            result = None

            try:
                # Execute with tools and stream chunks in real-time
                async for item in execute_with_tools_stream(
                    provider,
                    input_history,
                    self.system_prompt,
                    mcp_client=self.mcp_client,
                    tools=tools,
                ):
                    # Check if this is the final result
                    if isinstance(item, AgenticLoopResult):
                        result = item
                        continue

                    # This is a streaming chunk
                    chunk = item
                    any_yielded = True
                    chunk_type = chunk.get("type")
                    content = chunk.get("content", "")

                    if chunk_type == "text":
                        if content:
                            self.display_history[-1]["content"] += content

                    yield self.display_history, self.logic_history, chunk

                # Update history with delta
                if result is None:
                    logger.warning(
                        "execute_with_tools_stream did not yield AgenticLoopResult; "
                        "history not updated"
                    )
                elif mention == "all":
                    # For @all, extend logic_history with new entries
                    self.logic_history.extend(result.history_delta)
                else:
                    # For specific mention, input_history is self.logic_history
                    # execute_with_tools_stream no longer mutates history,
                    # so we extend it explicitly
                    input_history.extend(result.history_delta)

                if not any_yielded:
                    error_message = (
                        f"[System: {model_name.capitalize()}からの応答がありませんでした]"
                    )
                    self.display_history[-1]["content"] += error_message
                    new_entry = {
                        "role": model_name,
                        "content": [{"type": "text", "content": error_message}],
                    }
                    self.logic_history.append(new_entry)

            except (ValueError, Exception) as e:
                self._handle_api_error(e, model_name)
                yield self.display_history, self.logic_history, {"type": "error", "content": str(e)}

            yield self.display_history, self.logic_history, {"type": "text", "content": ""}

    def set_system_prompt(self, prompt):
        """Update system prompt

        Args:
            prompt: New system prompt text
        """
        self.system_prompt = prompt

    def append_tool_results(self, tool_results):
        """Append tool execution results to logic history.

        Args:
            tool_results: List of tool result dicts with name/content and optional tool_call_id.
        """
        if not tool_results:
            return

        content_parts = []
        for result in tool_results:
            if not isinstance(result, dict):
                logger.warning(
                    "Invalid tool result type: %s (expected dict), skipping",
                    type(result).__name__,
                )
                continue
            if result.get("type") == "tool_result":
                content_parts.append(result)
                continue
            content_parts.append(
                {
                    "type": "tool_result",
                    "name": result.get("name"),
                    "content": result.get("content", ""),
                    "tool_call_id": result.get("tool_call_id"),
                }
            )

        if content_parts:
            self.logic_history.append(
                {
                    "role": "tool",
                    "content": content_parts,
                }
            )
