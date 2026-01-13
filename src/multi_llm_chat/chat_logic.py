# Backward compatibility layer - delegates to new core and cli modules
import logging

from .core import (
    CHATGPT_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    call_chatgpt_api,
    call_gemini_api,
    format_history_for_chatgpt,
    format_history_for_gemini,
    list_gemini_models,
)
from .history import get_llm_response
from .history import reset_history as _reset_history
from .llm_provider import create_provider

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
            mcp_client: Optional MCP client for tool execution
        """
        self.display_history = display_history if display_history is not None else []
        self.logic_history = logic_history if logic_history is not None else []
        self.system_prompt = system_prompt

        # Store injected providers or None for lazy initialization
        self._gemini_provider = gemini_provider
        self._chatgpt_provider = chatgpt_provider
        self.mcp_client = mcp_client

    @property
    def gemini_provider(self):
        """Lazy-initialized Gemini provider"""
        if self._gemini_provider is None:
            self._gemini_provider = create_provider("gemini")
        # Add name attribute for execute_with_tools
        if not hasattr(self._gemini_provider, "name"):
            self._gemini_provider.name = "gemini"
        return self._gemini_provider

    @property
    def chatgpt_provider(self):
        """Lazy-initialized ChatGPT provider"""
        if self._chatgpt_provider is None:
            self._chatgpt_provider = create_provider("chatgpt")
        # Add name attribute for execute_with_tools
        if not hasattr(self._chatgpt_provider, "name"):
            self._chatgpt_provider.name = "chatgpt"
        return self._chatgpt_provider

    def _handle_api_error(self, error, provider_name):
        """Handle API errors in a consistent way

        Args:
            error: Exception that occurred
            provider_name: Name of the provider ("gemini" or "chatgpt")
        """
        provider_title = provider_name.capitalize()
        if isinstance(error, ValueError):
            # API key errors
            error_msg = f"[System: エラー - {str(error)}]"
        else:
            # Other API errors (network, blocked prompts, etc.)
            error_msg = f"[System: {provider_title} APIエラー - {str(error)}]"

        self.display_history[-1][1] = f"**{provider_title}:**\n{error_msg}"
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
        import copy

        from .core import execute_with_tools

        mention = parse_mention(user_message)

        # Add user message to histories (structured format)
        user_entry = {"role": "user", "content": [{"type": "text", "content": user_message}]}
        self.logic_history.append(user_entry)
        self.display_history.append([user_message, None])
        yield self.display_history, self.logic_history, {"type": "text", "content": ""}

        # If no mention, treat as memo (no LLM call)
        if mention is None:
            return

        # For @all, create snapshot so both LLMs see same history (deepcopy to avoid side effects)
        history_at_start = copy.deepcopy(self.logic_history)

        # Process models
        models_to_call = []
        if mention == "all":
            models_to_call = ["gemini", "chatgpt"]
        else:
            models_to_call = [mention]

        for model_name in models_to_call:
            provider = self.gemini_provider if model_name == "gemini" else self.chatgpt_provider
            label = ASSISTANT_LABELS[model_name]

            # For @all, we might need a new row in display_history for the second model
            if mention == "all" and model_name == "chatgpt":
                self.display_history.append([None, label])
            else:
                self.display_history[-1][1] = label

            yield self.display_history, self.logic_history, {"type": "text", "content": ""}

            # Prepare input history for this model
            if mention == "all":
                input_history = copy.deepcopy(history_at_start)
            else:
                input_history = self.logic_history

            any_yielded = False

            try:
                # Execute with tools and get result
                result = await execute_with_tools(
                    provider,
                    input_history,
                    self.system_prompt,
                    mcp_client=self.mcp_client,
                    tools=tools,
                )

                # Stream chunks from result (buffered streaming)
                for chunk in result.chunks:
                    any_yielded = True
                    chunk_type = chunk.get("type")
                    content = chunk.get("content", "")

                    if chunk_type == "text":
                        if content:
                            self.display_history[-1][1] += content

                    yield self.display_history, self.logic_history, chunk

                # Update history with delta
                if mention == "all":
                    # For @all, extend logic_history with new entries
                    self.logic_history.extend(result.history_delta)
                else:
                    # For specific mention, input_history is self.logic_history
                    # execute_with_tools no longer mutates history,
                    # so we extend it explicitly
                    input_history.extend(result.history_delta)

                if not any_yielded:
                    error_message = (
                        f"[System: {model_name.capitalize()}からの応答がありませんでした]"
                    )
                    self.display_history[-1][1] += error_message
                    new_entry = {
                        "role": model_name,
                        "content": [{"type": "text", "content": error_message}],
                    }
                    if mention == "all":
                        self.logic_history.append(new_entry)
                    else:
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
            self.logic_history.append({"role": "tool", "content": content_parts})


def main():
    """Backward compatible main function that returns only history"""
    import asyncio

    from .cli import main as _cli_main

    history, _system_prompt = asyncio.run(_cli_main())
    return history


def reset_history():
    """Clear conversation history (re-export for backward compatibility)."""
    return _reset_history()


__all__ = [
    "ChatService",
    "parse_mention",
    "main",
    "reset_history",
    "call_gemini_api",
    "call_chatgpt_api",
    "format_history_for_chatgpt",
    "format_history_for_gemini",
    "list_gemini_models",
    "get_llm_response",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_MODEL",
    "CHATGPT_MODEL",
]
