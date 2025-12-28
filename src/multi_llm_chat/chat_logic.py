# Backward compatibility layer - delegates to new core and cli modules
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

    Supports dependency injection of LLM providers for testing and session isolation.

    Attributes:
        display_history: UI-friendly history format [[user_msg, assistant_msg], ...]
        logic_history: API-friendly history format [{"role": "user", "content": "..."}]
        system_prompt: System prompt text for LLM context
        gemini_provider: Gemini LLM provider instance
        chatgpt_provider: ChatGPT LLM provider instance
    """

    def __init__(
        self,
        display_history=None,
        logic_history=None,
        system_prompt="",
        gemini_provider=None,
        chatgpt_provider=None,
    ):
        """Initialize ChatService with optional existing state and providers

        Args:
            display_history: Optional existing display history
            logic_history: Optional existing logic history
            system_prompt: Optional system prompt (default: "")
            gemini_provider: Optional Gemini provider instance (lazy-created if None)
            chatgpt_provider: Optional ChatGPT provider instance (lazy-created if None)
        """
        self.display_history = display_history if display_history is not None else []
        self.logic_history = logic_history if logic_history is not None else []
        self.system_prompt = system_prompt

        # Store injected providers or None for lazy initialization
        self._gemini_provider = gemini_provider
        self._chatgpt_provider = chatgpt_provider

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
        provider_title = provider_name.capitalize()
        if isinstance(error, ValueError):
            # API key errors
            error_msg = f"[System: エラー - {str(error)}]"
        else:
            # Other API errors (network, blocked prompts, etc.)
            error_msg = f"[System: {provider_title} APIエラー - {str(error)}]"

        self.display_history[-1][1] = f"**{provider_title}:**\n{error_msg}"
        self.logic_history.append({"role": provider_name, "content": error_msg})

    def process_message(self, user_message):
        """Process user message and generate LLM responses

        This is a generator function that yields intermediate states for streaming UI updates.

        Args:
            user_message: User's input message

        Yields:
            tuple: (display_history, logic_history) after each update
        """
        mention = parse_mention(user_message)

        # Add user message to histories
        self.logic_history.append({"role": "user", "content": user_message})
        self.display_history.append([user_message, None])
        yield self.display_history, self.logic_history

        # If no mention, treat as memo (no LLM call)
        if mention is None:
            return

        # For @all, create snapshot so both LLMs see same history
        history_snapshot = (
            [entry.copy() for entry in self.logic_history] if mention == "all" else None
        )

        # Process Gemini
        if mention in ["gemini", "all"]:
            gemini_label = ASSISTANT_LABELS["gemini"]
            self.display_history[-1][1] = gemini_label
            gemini_input_history = history_snapshot or self.logic_history

            try:
                # Use injected provider instance
                gemini_stream = self.gemini_provider.call_api(
                    gemini_input_history, self.system_prompt
                )

                full_response = ""
                for chunk in gemini_stream:
                    text = self.gemini_provider.extract_text_from_chunk(chunk)
                    if text:
                        full_response += text
                        self.display_history[-1][1] += text
                        yield self.display_history, self.logic_history

                self.logic_history.append({"role": "gemini", "content": full_response})
                if not full_response.strip():
                    self.display_history[-1][1] = (
                        "**Gemini:**\n[System: Geminiからの応答がありませんでした]"
                    )
            except (ValueError, Exception) as e:
                self._handle_api_error(e, "gemini")

            yield self.display_history, self.logic_history

        # Process ChatGPT
        if mention in ["chatgpt", "all"]:
            chatgpt_label = ASSISTANT_LABELS["chatgpt"]
            # For @all, add new display row to avoid prompt duplication
            if mention == "all":
                self.display_history.append([None, chatgpt_label])
            else:
                self.display_history[-1][1] = chatgpt_label

            chatgpt_input_history = history_snapshot or self.logic_history

            try:
                # Use injected provider instance
                chatgpt_stream = self.chatgpt_provider.call_api(
                    chatgpt_input_history, self.system_prompt
                )

                full_response = ""
                for chunk in chatgpt_stream:
                    text = self.chatgpt_provider.extract_text_from_chunk(chunk)
                    if text:
                        full_response += text
                        self.display_history[-1][1] += text
                        yield self.display_history, self.logic_history

                self.logic_history.append({"role": "chatgpt", "content": full_response})
                if not full_response.strip():
                    self.display_history[-1][1] = (
                        "**ChatGPT:**\n[System: ChatGPTからの応答がありませんでした]"
                    )
            except (ValueError, Exception) as e:
                self._handle_api_error(e, "chatgpt")

            yield self.display_history, self.logic_history

    def set_system_prompt(self, prompt):
        """Update system prompt

        Args:
            prompt: New system prompt text
        """
        self.system_prompt = prompt


def main():
    """Backward compatible main function that returns only history"""
    from .cli import main as _cli_main

    history, _system_prompt = _cli_main()
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
    "format_history_for_gemini",
    "format_history_for_chatgpt",
    "list_gemini_models",
    "get_llm_response",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_MODEL",
    "CHATGPT_MODEL",
]
