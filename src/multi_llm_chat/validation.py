from typing import Any, Callable, Dict, List, Optional

from .token_utils import get_max_context_length


def validate_system_prompt_length(
    system_prompt: str, model_name: str, token_calculator: Callable[[str, str], int] = None
) -> Dict[str, Any]:
    """Validate that system prompt doesn't exceed model's max context"""
    if not system_prompt:
        return {"valid": True}

    if token_calculator is None:
        raise ValueError("token_calculator is required")

    max_length = get_max_context_length(model_name)
    prompt_tokens = token_calculator(system_prompt, model_name)

    if prompt_tokens > max_length:
        return {
            "valid": False,
            "error": (
                f"System prompt ({prompt_tokens} tokens) exceeds "
                f"max context length ({max_length} tokens)"
            ),
        }

    return {"valid": True}


def validate_context_length(
    history: List[Dict[str, Any]],
    system_prompt: Optional[str],
    model_name: str,
    token_calculator: Callable[[str, str], int] = None,
) -> Dict[str, Any]:
    """Validate that system prompt + latest turn doesn't exceed max context"""
    max_length = get_max_context_length(model_name)

    if token_calculator is None:
        raise ValueError("token_calculator is required")

    # Calculate system prompt tokens
    system_tokens = 0
    if system_prompt:
        system_tokens = token_calculator(system_prompt, model_name)

    # Get latest turn (may be just user message, or user + assistant)
    if not history:
        # Only system prompt
        if system_tokens > max_length:
            return {
                "valid": False,
                "error": (
                    f"System prompt alone ({system_tokens} tokens) exceeds "
                    f"max context ({max_length} tokens)"
                ),
            }
        return {"valid": True}

    # Calculate tokens for latest turn (user + all assistant responses)
    latest_tokens = 0
    if history:
        # Find the last user message
        i = len(history) - 1
        while i >= 0 and history[i]["role"] in ["gemini", "chatgpt"]:
            # Add assistant message tokens
            content = history[i].get("content", "")
            latest_tokens += token_calculator(content, model_name)
            i -= 1

        # Add user message if found
        if i >= 0 and history[i]["role"] == "user":
            content = history[i].get("content", "")
            latest_tokens += token_calculator(content, model_name)

    total_tokens = system_tokens + latest_tokens

    if total_tokens > max_length:
        return {
            "valid": False,
            "error": (
                f"Single turn too long: {total_tokens} tokens exceeds "
                f"max context ({max_length} tokens)"
            ),
        }

    return {"valid": True}
