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
from .history import reset_history as _reset_history


def get_llm_response(history, index):
    """指定インデックスのLLM応答本文を取得する（最新が0）。"""
    if index < 0:
        raise IndexError("index must be non-negative")

    # 最新のLLM応答から順に拾う
    responses = [
        entry.get("content", "")
        for entry in reversed(history or [])
        if entry.get("role") in {"gemini", "chatgpt"}
    ]

    try:
        return responses[index]
    except IndexError as exc:
        raise IndexError("LLM response not found for the given index") from exc


def main():
    """Backward compatible main function that returns only history"""
    history, _system_prompt = _cli_main()
    return history


def reset_history():
    """Clear conversation history (re-export for backward compatibility)."""
    return _reset_history()


__all__ = [
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
