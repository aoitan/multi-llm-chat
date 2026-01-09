import os
import google.generativeai as genai
import openai
from dotenv import load_dotenv

from .history_utils import (
    LLM_ROLES, 
    get_provider_name_from_model as _get_provider_name_from_model, 
    prepare_request
)
from .token_utils import (
    estimate_tokens as _estimate_tokens,
    get_token_info,
    get_max_context_length,
    calculate_tokens
)
from .compression import (
    prune_history_sliding_window,
    get_pruning_info
)
from .validation import (
    validate_system_prompt_length,
    validate_context_length
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

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use llm_provider.get_provider("gemini") instead.
    """
    from .llm_provider import get_provider

    try:
        provider = get_provider("gemini")
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

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use llm_provider.get_provider("chatgpt") instead.
    """
    from .llm_provider import get_provider

    try:
        provider = get_provider("chatgpt")
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

def extract_text_from_chunk(chunk, model_name):
    """Extract text content from API response chunk

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.extract_text_from_chunk() directly.
    """
    from .llm_provider import get_provider

    try:
        provider_name = _get_provider_name_from_model(model_name)
        provider = get_provider(provider_name)
        return provider.extract_text_from_chunk(chunk)
    except Exception:
        if isinstance(chunk, str):
            return chunk
        return ""