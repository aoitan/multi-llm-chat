from typing import List, Dict, Any, Optional, Callable
from .history_utils import get_provider_name_from_model


def prune_history_sliding_window(
    history: List[Dict[str, Any]],
    max_tokens: int,
    model_name: str,
    system_prompt: Optional[str] = None,
    token_calculator: Callable[[str, str], int] = None,
) -> List[Dict[str, Any]]:
    """Prune conversation history using sliding window approach"""
    if not history:
        return history

    if token_calculator is None:
        raise ValueError("token_calculator is required")

    # Calculate system prompt tokens
    system_tokens = 0
    if system_prompt:
        system_tokens = token_calculator(system_prompt, model_name)

    # Calculate tokens for each entry
    entry_tokens = []
    for entry in history:
        content = entry.get("content", "")
        tokens = token_calculator(content, model_name)
        entry_tokens.append(tokens)

    # Calculate total tokens
    total_tokens = system_tokens + sum(entry_tokens)

    # If within limit, return as-is
    if total_tokens <= max_tokens:
        return history

    # Prune from the beginning, preserving complete user-assistant turns
    pruned_history = []
    accumulated_tokens = system_tokens

    # Start from the end and work backwards
    i = len(history) - 1
    while i >= 0:
        entry = history[i]
        role = entry["role"]

        if role in ["gemini", "chatgpt"]:
            # Collect all consecutive assistant messages (for @all pattern)
            assistant_messages = []
            assistant_tokens = 0
            j = i

            while j >= 0 and history[j]["role"] in ["gemini", "chatgpt"]:
                assistant_messages.insert(0, history[j])
                assistant_tokens += entry_tokens[j]
                j -= 1

            # Check if there's a user message before the assistants
            if j >= 0 and history[j]["role"] == "user":
                # Calculate cost of entire turn (user + all assistants)
                turn_tokens = entry_tokens[j] + assistant_tokens

                if accumulated_tokens + turn_tokens <= max_tokens:
                    # Add entire turn (user + all assistants)
                    pruned_history.insert(0, history[j])  # User message
                    for msg in assistant_messages:
                        # Append assistants in order
                        pruned_history.insert(len(pruned_history), msg)
                    accumulated_tokens += turn_tokens
                    i = j - 1  # Skip to before user message
                else:
                    # Turn doesn't fit, stop here
                    break
            else:
                # Orphaned assistant messages (no preceding user) - skip them
                i = j
        elif role == "user":
            # Standalone user message (no assistant response yet)
            if accumulated_tokens + entry_tokens[i] <= max_tokens:
                pruned_history.insert(0, entry)
                accumulated_tokens += entry_tokens[i]
            i -= 1
        else:
            # Unknown role - skip
            i -= 1

    return pruned_history


def get_pruning_info(
    history: List[Dict[str, Any]],
    max_tokens: int,
    model_name: str,
    system_prompt: Optional[str] = None,
    token_calculator: Callable[[str, str], int] = None,
) -> Dict[str, Any]:
    """Get information about how history would be pruned"""
    if not history:
        return {
            "turns_to_remove": 0,
            "original_length": 0,
            "pruned_length": 0,
        }

    if token_calculator is None:
        raise ValueError("token_calculator is required")

    # Calculate current total
    system_tokens = 0
    if system_prompt:
        system_tokens = token_calculator(system_prompt, model_name)

    original_tokens = system_tokens
    for entry in history:
        content = entry.get("content", "")
        original_tokens += token_calculator(content, model_name)

    # Get pruned version
    pruned = prune_history_sliding_window(
        history, max_tokens, model_name, system_prompt, token_calculator
    )

    pruned_tokens = system_tokens
    for entry in pruned:
        content = entry.get("content", "")
        pruned_tokens += token_calculator(content, model_name)

    turns_removed = len(history) - len(pruned)

    return {
        "turns_to_remove": turns_removed,
        "original_length": original_tokens,
        "pruned_length": pruned_tokens,
    }
