"""Legacy API module - DEPRECATED backward compatibility wrappers

This module contains deprecated wrapper functions that provide backward compatibility
for code using the old core.py API. New code should use the underlying implementations directly:

- Provider layer: llm_provider.GeminiProvider, llm_provider.ChatGPTProvider
- History utilities: history_utils.prepare_request()
- Direct provider usage: provider = create_provider('gemini'); provider.call_api(...)

All functions in this module are marked as DEPRECATED and may be removed in future versions.
"""

import asyncio
import logging
import warnings
from typing import Any, Dict, List, Optional

import openai

from ..history_utils import (
    get_provider_name_from_model as _get_provider_name_from_model,
)
from ..history_utils import (
    prepare_request as _prepare_request,
)
from ..llm_provider import (
    ChatGPTProvider,
    GeminiProvider,
    create_provider,
)

logger = logging.getLogger(__name__)


def load_api_key(env_var_name: str) -> str:
    """Load API key from environment

    DEPRECATED: This is a backward compatibility wrapper.
    New code should import from llm_provider directly.

    Args:
        env_var_name: Name of the environment variable (e.g., 'GOOGLE_API_KEY')

    Returns:
        str: API key value or None if not found
    """
    from ..llm_provider import load_api_key as _load_api_key

    return _load_api_key(env_var_name)


def format_history_for_gemini(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert history to Gemini API format

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use GeminiProvider.format_history() directly.

    Args:
        history: Conversation history

    Returns:
        List[Dict]: Gemini-formatted history
    """
    return GeminiProvider.format_history(history)


def format_history_for_chatgpt(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert history to ChatGPT API format

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use ChatGPTProvider.format_history() directly.

    Args:
        history: Conversation history

    Returns:
        List[Dict]: ChatGPT-formatted history
    """
    return ChatGPTProvider.format_history(history)


def extract_text_from_chunk(chunk: Any, model_name: str) -> str:
    """Extract text content from API response chunk

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.extract_text_from_chunk() directly.

    Args:
        chunk: API response chunk
        model_name: Name of the model (used to determine provider)

    Returns:
        str: Extracted text content
    """
    try:
        provider_name = _get_provider_name_from_model(model_name)
        provider = create_provider(provider_name)
        return provider.extract_text_from_chunk(chunk)
    except Exception:
        if isinstance(chunk, str):
            return chunk
        return ""


def prepare_request(history: List[Dict[str, Any]], system_prompt: Optional[str], model_name: str):
    """Prepare API request with system prompt and history

    DEPRECATED: This is a backward compatibility wrapper.
    New code should import from history_utils directly.

    Args:
        history: Conversation history
        system_prompt: System prompt (optional)
        model_name: Name of the model

    Returns:
        Prepared request (format depends on provider)
    """
    return _prepare_request(history, system_prompt, model_name)


async def call_gemini_api_async(history: List[Dict[str, Any]], system_prompt: Optional[str] = None):
    """Call Gemini API asynchronously.

    DEPRECATED: Use ChatService or create_provider('gemini').call_api() instead.

    Args:
        history: Conversation history
        system_prompt: System prompt (optional)

    Yields:
        API response chunks
    """
    try:
        provider = create_provider("gemini")
        async for chunk in provider.call_api(history, system_prompt):
            yield chunk
    except ValueError as e:
        yield f"Gemini API Error: {e}"
    except Exception as e:
        error_msg = f"Gemini API Error: An unexpected error occurred: {e}"
        print(error_msg)
        # Note: list_gemini_models() is in providers_facade.py, not imported here
        # to avoid circular dependency
        yield error_msg


def call_gemini_api(history: List[Dict[str, Any]], system_prompt: Optional[str] = None):
    """Call Gemini API (synchronous wrapper for backward compatibility)

    DEPRECATED: Use ChatService or create_provider('gemini').call_api() instead.

    Args:
        history: Conversation history
        system_prompt: System prompt (optional)

    Yields:
        API response chunks
    """
    warnings.warn(
        "call_gemini_api() is deprecated and will be removed. "
        "Use ChatService or create_provider() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_gen = call_gemini_api_async(history, system_prompt)
    while True:
        try:
            yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            break


async def call_chatgpt_api_async(
    history: List[Dict[str, Any]], system_prompt: Optional[str] = None
):
    """Call ChatGPT API asynchronously.

    DEPRECATED: Use ChatService or create_provider('chatgpt').call_api() instead.

    Args:
        history: Conversation history
        system_prompt: System prompt (optional)

    Yields:
        API response chunks
    """
    try:
        provider = create_provider("chatgpt")
        async for chunk in provider.call_api(history, system_prompt):
            yield chunk
    except ValueError as e:
        yield f"ChatGPT API Error: {e}"
    except openai.APIError as e:
        yield f"ChatGPT API Error: OpenAI APIからエラーが返されました: {e}"
    except openai.APITimeoutError as e:
        yield f"ChatGPT API Error: リクエストがタイムアウトしました: {e}"
    except openai.APIConnectionError as e:
        yield f"ChatGPT API Error: APIへの接続に失敗しました: {e}"
    except Exception as e:
        yield f"ChatGPT API Error: 予期せぬエラーが発生しました: {e}"


def call_chatgpt_api(history: List[Dict[str, Any]], system_prompt: Optional[str] = None):
    """Call ChatGPT API (synchronous wrapper for backward compatibility)

    DEPRECATED: Use ChatService or create_provider('chatgpt').call_api() instead.

    Args:
        history: Conversation history
        system_prompt: System prompt (optional)

    Yields:
        API response chunks
    """
    warnings.warn(
        "call_chatgpt_api() is deprecated and will be removed. "
        "Use ChatService or create_provider() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_gen = call_chatgpt_api_async(history, system_prompt)
    while True:
        try:
            yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            break


async def stream_text_events_async(
    history: List[Dict[str, Any]], provider_name: str, system_prompt: Optional[str] = None
):
    """Stream normalized text events from a provider asynchronously.

    DEPRECATED: Use provider.stream_text_events() directly.

    Args:
        history: Conversation history
        provider_name: Name of the provider ('gemini' or 'chatgpt')
        system_prompt: System prompt (optional)

    Yields:
        Normalized text event chunks
    """
    provider = create_provider(provider_name)
    async for chunk in provider.stream_text_events(history, system_prompt):
        yield chunk


def stream_text_events(
    history: List[Dict[str, Any]], provider_name: str, system_prompt: Optional[str] = None
):
    """Stream normalized text events (synchronous wrapper for backward compatibility)

    DEPRECATED: Use provider.stream_text_events() directly.

    Args:
        history: Conversation history
        provider_name: Name of the provider ('gemini' or 'chatgpt')
        system_prompt: System prompt (optional)

    Yields:
        Normalized text event chunks
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_gen = stream_text_events_async(history, provider_name, system_prompt)
    while True:
        try:
            yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            break
