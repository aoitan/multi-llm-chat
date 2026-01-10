import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import openai
from dotenv import load_dotenv

from .compression import (
    get_pruning_info as _get_pruning_info,
)
from .compression import (
    prune_history_sliding_window as _prune_history_sliding_window,
)
from .history_utils import (
    LLM_ROLES as LLM_ROLES,
)
from .history_utils import (
    get_provider_name_from_model as _get_provider_name_from_model,
)
from .history_utils import (
    prepare_request as _prepare_request,
)
from .token_utils import (
    estimate_tokens as _estimate_tokens_impl,
)
from .token_utils import (
    get_max_context_length as _get_max_context_length,
)
from .validation import (
    validate_context_length as _validate_context_length,
)
from .validation import (
    validate_system_prompt_length as _validate_system_prompt_length,
)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo")


def load_api_key(env_var_name):
    """Load API key from environment"""
    return os.getenv(env_var_name)


def format_history_for_gemini(history):
    """Convert history to Gemini API format

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use GeminiProvider.format_history() directly.
    """
    from .llm_provider import GeminiProvider

    return GeminiProvider.format_history(history)


def format_history_for_chatgpt(history):
    """Convert history to ChatGPT API format

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use ChatGPTProvider.format_history() directly.
    """
    from .llm_provider import ChatGPTProvider

    return ChatGPTProvider.format_history(history)


def list_gemini_models():
    """List available Gemini models"""
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
        return

    genai.configure(api_key=GOOGLE_API_KEY)
    print("利用可能なGeminiモデル:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  - {m.name}")


def call_gemini_api(history, system_prompt=None):
    """Call Gemini API with optional system prompt

    DEPRECATED (Will be removed in future version):
    This function uses a global shared provider instance which causes
    prompt pollution in concurrent environments (e.g., Gradio sessions).

    **DO NOT USE in production code.**

    Migration path:
    - For session-scoped: ChatService with injected providers
    - For one-off calls: llm_provider.create_provider("gemini")

    SECURITY WARNING: Using this in multi-user environments WILL result in:
    - System prompt leaking between users
    - Cached models being shared across sessions
    - Race conditions in concurrent requests

    This function is kept only for backward compatibility with existing tests.
    """
    import warnings

    warnings.warn(
        "call_gemini_api() is deprecated and will be removed. "
        "Use ChatService or create_provider() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from multi_llm_chat.llm_provider import create_provider

    try:
        provider = create_provider("gemini")
        yield from provider.call_api(history, system_prompt)
    except ValueError as e:
        yield f"Gemini API Error: {e}"
    except Exception as e:
        error_msg = f"Gemini API Error: An unexpected error occurred: {e}"
        print(error_msg)
        try:
            list_gemini_models()
        except Exception:
            pass
        yield error_msg


def call_chatgpt_api(history, system_prompt=None):
    """Call ChatGPT API with optional system prompt

    DEPRECATED (Will be removed in future version):
    This function uses a global shared provider instance which causes
    prompt pollution in concurrent environments (e.g., Gradio sessions).

    **DO NOT USE in production code.**

    Migration path:
    - For session-scoped: ChatService with injected providers
    - For one-off calls: llm_provider.create_provider("chatgpt")

    SECURITY WARNING: Using this in multi-user environments WILL result in:
    - System prompt leaking between users
    - Client instances being shared across sessions
    - Race conditions in concurrent requests

    This function is kept only for backward compatibility with existing tests.
    """
    import warnings

    warnings.warn(
        "call_chatgpt_api() is deprecated and will be removed. "
        "Use ChatService or create_provider() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from multi_llm_chat.llm_provider import create_provider

    try:
        provider = create_provider("chatgpt")
        yield from provider.call_api(history, system_prompt)
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


def stream_text_events(history, provider_name, system_prompt=None):
    """Stream normalized text events from a provider.

    Args:
        history: Conversation history for the request
        provider_name: Provider identifier ("gemini" or "chatgpt")
        system_prompt: Optional system instruction

    Yields:
        str: Normalized text chunks
    """
    from multi_llm_chat.llm_provider import create_provider

    provider = create_provider(provider_name)
    yield from provider.stream_text_events(history, system_prompt)


def extract_text_from_chunk(chunk, model_name):
    """Extract text content from API response chunk

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.extract_text_from_chunk() directly.
    """
    from multi_llm_chat.llm_provider import create_provider

    try:
        provider_name = _get_provider_name_from_model(model_name)
        provider = create_provider(provider_name)
        return provider.extract_text_from_chunk(chunk)
    except Exception:
        if isinstance(chunk, str):
            return chunk
        return ""


def prepare_request(history, system_prompt, model_name):
    """Prepare API request with system prompt and history"""
    return _prepare_request(history, system_prompt, model_name)


def _estimate_tokens(text):
    return _estimate_tokens_impl(text)


def get_max_context_length(model_name):
    return _get_max_context_length(model_name)


def calculate_tokens(text: str, model_name: str) -> int:
    """Calculate token count for text using model-appropriate method"""

    from .llm_provider import get_provider

    provider_name = _get_provider_name_from_model(model_name)

    provider = get_provider(provider_name)

    result = provider.get_token_info(text, history=None, model_name=model_name)

    return result["input_tokens"]


def get_token_info(
    text: str, model_name: str, history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Get token information for the given text and model

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.get_token_info() directly.
    """

    from .llm_provider import TIKTOKEN_AVAILABLE, get_provider

    # Determine provider from model name
    provider_name = _get_provider_name_from_model(model_name)

    # Get provider and delegate token calculation with actual model name
    provider = get_provider(provider_name)

    result = provider.get_token_info(text, history, model_name=model_name)

    # Add is_estimated flag for backward compatibility
    is_estimated = provider_name == "gemini" or not TIKTOKEN_AVAILABLE

    return {
        "token_count": result["input_tokens"],
        "max_context_length": result["max_tokens"],
        "is_estimated": is_estimated,
    }


def prune_history_sliding_window(history, max_tokens, model_name, system_prompt=None):
    """Prune conversation history using sliding window approach"""

    return _prune_history_sliding_window(
        history, max_tokens, model_name, system_prompt, token_calculator=calculate_tokens
    )


def get_pruning_info(history, max_tokens, model_name, system_prompt=None):
    """Get information about how history would be pruned"""

    return _get_pruning_info(
        history, max_tokens, model_name, system_prompt, token_calculator=calculate_tokens
    )


def validate_system_prompt_length(system_prompt, model_name):
    """Validate that system prompt doesn't exceed model's max context"""

    return _validate_system_prompt_length(
        system_prompt, model_name, token_calculator=calculate_tokens
    )


def validate_context_length(history, system_prompt, model_name):
    """Validate that system prompt + latest turn doesn't exceed max context"""

    return _validate_context_length(
        history, system_prompt, model_name, token_calculator=calculate_tokens
    )
