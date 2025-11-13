# Backward compatibility layer - delegates to new core and cli modules
from .cli import main as _cli_main
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


def main():
    """Backward compatible main function that returns only history"""
    history, _system_prompt = _cli_main()
    return history


__all__ = [
    "main",
    "call_gemini_api",
    "call_chatgpt_api",
    "format_history_for_gemini",
    "format_history_for_chatgpt",
    "list_gemini_models",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_MODEL",
    "CHATGPT_MODEL",
]
