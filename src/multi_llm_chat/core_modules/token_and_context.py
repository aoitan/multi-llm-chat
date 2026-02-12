"""Token and context management module

This module provides wrapper functions for token calculation, context validation,
and history pruning. These are primarily wrappers around implementations in:
- token_utils: estimate_tokens, get_max_context_length
- validation: validate_system_prompt_length, validate_context_length
- compression: prune_history_sliding_window, get_pruning_info

New code should import directly from those modules or use Provider.get_token_info().
"""

from typing import Any, Dict, List, Optional

from ..compression import (
    get_pruning_info as _get_pruning_info,
)
from ..compression import (
    prune_history_sliding_window as _prune_history_sliding_window,
)
from ..history_utils import get_provider_name_from_model as _get_provider_name_from_model
from ..llm_provider import create_provider
from ..token_utils import (
    estimate_tokens as _estimate_tokens_impl,
)
from ..token_utils import (
    get_max_context_length as _get_max_context_length,
)
from ..validation import (
    validate_context_length as _validate_context_length,
)
from ..validation import (
    validate_system_prompt_length as _validate_system_prompt_length,
)


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text

    Internal wrapper for backward compatibility.
    New code should use token_utils.estimate_tokens() directly.

    Args:
        text: Text to estimate tokens for

    Returns:
        int: Estimated token count
    """
    return _estimate_tokens_impl(text)


def get_max_context_length(model_name: str) -> int:
    """Get maximum context length for model

    Wrapper for token_utils.get_max_context_length().

    Args:
        model_name: Name of the model

    Returns:
        int: Maximum context length in tokens
    """
    return _get_max_context_length(model_name)


def calculate_tokens(text: str, model_name: str) -> int:
    """Calculate token count for text using model-appropriate method

    Args:
        text: Text to calculate tokens for
        model_name: Name of the model

    Returns:
        int: Token count
    """
    provider_name = _get_provider_name_from_model(model_name)
    provider = create_provider(provider_name)  # Issue #116: Use create_provider

    result = provider.get_token_info(text, history=None, model_name=model_name)

    return result["input_tokens"]


def get_token_info(
    text: str, model_name: str, history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Get token information for the given text and model

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.get_token_info() directly.

    Args:
        text: Text to get token info for
        model_name: Name of the model
        history: Optional conversation history

    Returns:
        Dict with keys: token_count, max_context_length, is_estimated
    """
    from ..llm_provider import TIKTOKEN_AVAILABLE

    # Determine provider from model name
    provider_name = _get_provider_name_from_model(model_name)

    # Get provider and delegate token calculation with actual model name
    provider = create_provider(provider_name)  # Issue #116: Use create_provider

    result = provider.get_token_info(text, history, model_name=model_name)

    # Add is_estimated flag for backward compatibility
    is_estimated = provider_name == "gemini" or not TIKTOKEN_AVAILABLE

    return {
        "token_count": result["input_tokens"],
        "max_context_length": result["max_tokens"],
        "is_estimated": is_estimated,
    }


def prune_history_sliding_window(
    history: List[Dict[str, Any]],
    max_tokens: int,
    model_name: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Prune conversation history using sliding window approach

    Wrapper for compression.prune_history_sliding_window().

    Args:
        history: Conversation history
        max_tokens: Maximum tokens to keep
        model_name: Name of the model
        system_prompt: Optional system prompt

    Returns:
        List[Dict]: Pruned history
    """
    return _prune_history_sliding_window(
        history, max_tokens, model_name, system_prompt, token_calculator=calculate_tokens
    )


def get_pruning_info(
    history: List[Dict[str, Any]],
    max_tokens: int,
    model_name: str,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Get information about how history would be pruned

    Wrapper for compression.get_pruning_info().

    Args:
        history: Conversation history
        max_tokens: Maximum tokens to keep
        model_name: Name of the model
        system_prompt: Optional system prompt

    Returns:
        Dict with pruning information
    """
    return _get_pruning_info(
        history, max_tokens, model_name, system_prompt, token_calculator=calculate_tokens
    )


def validate_system_prompt_length(system_prompt: str, model_name: str) -> Dict[str, Any]:
    """Validate that system prompt doesn't exceed model's max context

    Wrapper for validation.validate_system_prompt_length().

    Args:
        system_prompt: System prompt to validate
        model_name: Name of the model

    Returns:
        Dict with validation result (valid: bool, error: str if invalid)
    """
    return _validate_system_prompt_length(
        system_prompt, model_name, token_calculator=calculate_tokens
    )


def validate_context_length(
    history: List[Dict[str, Any]], system_prompt: Optional[str], model_name: str
) -> Dict[str, Any]:
    """Validate that system prompt + latest turn doesn't exceed max context

    Wrapper for validation.validate_context_length().

    Args:
        history: Conversation history
        system_prompt: Optional system prompt
        model_name: Name of the model

    Returns:
        Dict with validation result (valid: bool, error: str if invalid)
    """
    return _validate_context_length(
        history, system_prompt, model_name, token_calculator=calculate_tokens
    )
