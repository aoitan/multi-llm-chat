# Backward compatibility layer - delegates to new core and cli modules
from .core import (
    CHATGPT_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    format_history_for_chatgpt,
    format_history_for_gemini,
    list_gemini_models,
)
from .history import get_llm_response
from .history import reset_history as _reset_history
from .llm_provider import get_provider


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


class ChatService:
    """Business logic layer for chat operations

    Encapsulates core chat logic (mention parsing, LLM routing, history management)
    independent of UI implementation. This allows both CLI and WebUI to share
    the same business logic without duplication.

    Attributes:
        display_history: UI-friendly history format [[user_msg, assistant_msg], ...]
        logic_history: API-friendly history format [{"role": "user", "content": "..."}]
        system_prompt: System prompt text for LLM context
    """

    def __init__(self, display_history=None, logic_history=None, system_prompt=""):
        """Initialize ChatService with optional existing state

        Args:
            display_history: Optional existing display history
            logic_history: Optional existing logic history
            system_prompt: Optional system prompt (default: "")
        """
        self.display_history = display_history if display_history is not None else []
        self.logic_history = logic_history if logic_history is not None else []
        self.system_prompt = system_prompt

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
            self.display_history[-1][1] = "**Gemini:**\n"
            gemini_input_history = history_snapshot or self.logic_history
            
            # Use provider abstraction
            provider = get_provider("gemini")
            gemini_stream = provider.call_api(gemini_input_history, self.system_prompt)

            full_response = ""
            for chunk in gemini_stream:
                text = provider.extract_text_from_chunk(chunk)
                if text:
                    full_response += text
                    self.display_history[-1][1] += text
                    yield self.display_history, self.logic_history

            self.logic_history.append({"role": "gemini", "content": full_response})
            if not full_response.strip():
                self.display_history[-1][1] = (
                    "**Gemini:**\n[System: Geminiからの応答がありませんでした]"
                )
            yield self.display_history, self.logic_history

        # Process ChatGPT
        if mention in ["chatgpt", "all"]:
            # For @all, add new display row to avoid prompt duplication
            if mention == "all":
                self.display_history.append([None, "**ChatGPT:**\n"])
            else:
                self.display_history[-1][1] = "**ChatGPT:**\n"

            chatgpt_input_history = history_snapshot or self.logic_history
            
            # Use provider abstraction
            provider = get_provider("chatgpt")
            chatgpt_stream = provider.call_api(chatgpt_input_history, self.system_prompt)

            full_response = ""
            for chunk in chatgpt_stream:
                text = provider.extract_text_from_chunk(chunk)
                if text:
                    full_response += text
                    self.display_history[-1][1] += text
                    yield self.display_history, self.logic_history

            self.logic_history.append({"role": "chatgpt", "content": full_response})
            if not full_response.strip():
                self.display_history[-1][1] = (
                    "**ChatGPT:**\n[System: ChatGPTからの応答がありませんでした]"
                )
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
