import os
from collections import OrderedDict

import google.generativeai as genai
import openai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo")

_gemini_model = None
_gemini_models_cache = OrderedDict()  # LRU cache for models with system prompts
_gemini_cache_max_size = 10  # Limit cache size to prevent memory leak
_openai_client = None


def load_api_key(env_var_name):
    """Load API key from environment"""
    return os.getenv(env_var_name)


def _configure_gemini():
    """Configure the Gemini SDK if an API key is available."""
    if not GOOGLE_API_KEY:
        return False
    genai.configure(api_key=GOOGLE_API_KEY)
    return True


def _get_gemini_model(system_prompt=None):
    """Get or create a cached Gemini model instance

    Args:
        system_prompt: Optional system instruction for the model

    Returns:
        GenerativeModel instance or None if API key not available
    """
    global _gemini_model, _gemini_models_cache, _gemini_cache_max_size

    if not _configure_gemini():
        return None

    # If no system prompt, use the default cached model
    if system_prompt is None:
        if _gemini_model is None:
            _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        return _gemini_model

    # For system prompts, use LRU cache
    if system_prompt in _gemini_models_cache:
        # Move to end (most recently used)
        _gemini_models_cache.move_to_end(system_prompt)
        return _gemini_models_cache[system_prompt]

    # Create new model and add to cache
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
    _gemini_models_cache[system_prompt] = model

    # Evict oldest if cache is full
    if len(_gemini_models_cache) > _gemini_cache_max_size:
        _gemini_models_cache.popitem(last=False)

    return model


def _get_openai_client():
    """Return a cached OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if not OPENAI_API_KEY:
        return None
    _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def format_history_for_gemini(history):
    """Convert history to Gemini API format"""
    gemini_history = []
    for entry in history:
        role = "user" if entry["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [entry["content"]]})
    return gemini_history


def format_history_for_chatgpt(history):
    """Convert history to ChatGPT API format"""
    chatgpt_history = []
    for entry in history:
        if entry["role"] == "system":
            chatgpt_history.append({"role": "system", "content": entry["content"]})
            continue
        role = "user" if entry["role"] == "user" else "assistant"
        chatgpt_history.append({"role": role, "content": entry["content"]})
    return chatgpt_history


def get_token_info(text, model_name, history=None):
    """Get token information for the given text and model

    Args:
        text: System prompt text
        model_name: Model to use for context length calculation
        history: Optional conversation history to include in token count

    Returns:
        dict with token_count, max_context_length, and is_estimated
    """
    # Simple estimation based on character count (4 chars ≈ 1 token)
    estimated_tokens = len(text) // 4

    # Add history tokens if provided
    if history:
        history_text = "".join(entry.get("content", "") for entry in history)
        estimated_tokens += len(history_text) // 4

    # Define max context length per model (updated for GPT-4 variants)
    model_lower = model_name.lower()
    if "gemini" in model_lower:
        max_context = 1048576
    elif "gpt-4o" in model_lower:
        max_context = 128000
    elif "gpt-4-turbo" in model_lower or "gpt-4-1106" in model_lower:
        max_context = 128000
    elif "gpt-4" in model_lower:
        # Default GPT-4 (non-turbo, non-1106) has 8K context
        max_context = 8192
    elif "gpt-3.5-turbo-16k" in model_lower:
        max_context = 16385
    elif "gpt-3.5" in model_lower:
        max_context = 4096
    else:
        # Conservative default
        max_context = 4096

    return {
        "token_count": estimated_tokens,
        "max_context_length": max_context,
        "is_estimated": True,
    }


def prepare_request(history, system_prompt, model_name):
    """Prepare API request with system prompt and history"""
    if "gemini" in model_name.lower():
        # For Gemini, return tuple (system_prompt, history)
        if system_prompt:
            return (system_prompt, history)
        else:
            return (None, history)
    else:
        # For OpenAI-compatible models, add system message to history
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + history
        else:
            return history


def list_gemini_models():
    """List available Gemini models"""
    if not _configure_gemini():
        print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
        return

    print("利用可能なGeminiモデル:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  - {m.name}")


def call_gemini_api(history, system_prompt=None):
    """Call Gemini API with optional system prompt"""
    # Use cached model with system prompt
    model = _get_gemini_model(system_prompt)

    if not model:
        yield ("Gemini API Error: GOOGLE_API_KEY not found in environment variables or .env file.")
        return

    try:
        gemini_history = format_history_for_gemini(history)
        response_stream = model.generate_content(gemini_history, stream=True)
        for chunk in response_stream:
            yield chunk
    except genai.types.BlockedPromptException as e:
        yield f"Gemini API Error: Prompt was blocked due to safety concerns. Details: {e}"
    except Exception as e:
        print(f"Gemini API Error: An unexpected error occurred: {e}")
        list_gemini_models()
        yield f"Gemini API Error: An unexpected error occurred: {e}"


def call_chatgpt_api(history, system_prompt=None):
    """Call ChatGPT API with optional system prompt"""
    try:
        client = _get_openai_client()
        if client is None:
            yield "ChatGPT API Error: OPENAI_API_KEYが設定されていません。"
            return

        # Use prepare_request to create the base history
        prepared_history = prepare_request(history, system_prompt, "chatgpt")
        # Format the entire history for the API
        formatted_history = format_history_for_chatgpt(prepared_history)

        response_stream = client.chat.completions.create(
            model=CHATGPT_MODEL, messages=formatted_history, stream=True
        )
        for chunk in response_stream:
            yield chunk
    except openai.APIError as e:
        yield f"ChatGPT API Error: OpenAI APIからエラーが返されました: {e}"
    except openai.APITimeoutError as e:
        yield f"ChatGPT API Error: リクエストがタイムアウトしました: {e}"
    except openai.APIConnectionError as e:
        yield f"ChatGPT API Error: APIへの接続に失敗しました: {e}"
    except Exception as e:
        yield f"ChatGPT API Error: 予期せぬエラーが発生しました: {e}"
