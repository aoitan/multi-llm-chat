import hashlib
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
_gemini_models_cache = OrderedDict()  # LRU cache: hash -> (prompt, model)
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


def _hash_prompt(prompt):
    """Generate SHA256 hash for a prompt to use as cache key

    Args:
        prompt: System prompt text

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


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
    if not system_prompt or not system_prompt.strip():
        if _gemini_model is None:
            _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        return _gemini_model

    # For system prompts, use LRU cache with hash key
    prompt_hash = _hash_prompt(system_prompt)

    if prompt_hash in _gemini_models_cache:
        # Verify prompt hasn't changed (hash collision check)
        cached_prompt, cached_model = _gemini_models_cache[prompt_hash]
        if cached_prompt == system_prompt:
            # Move to end (most recently used)
            _gemini_models_cache.move_to_end(prompt_hash)
            return cached_model
        # Hash collision - evict and recreate

    # Create new model and add to cache
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
    _gemini_models_cache[prompt_hash] = (system_prompt, model)

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
    """Convert history to Gemini API format

    Filters out responses from other LLMs (e.g., ChatGPT) to avoid
    sending Gemini messages it didn't generate, which would create
    a self-contradictory conversation.
    """
    gemini_history = []
    for entry in history:
        role = entry["role"]
        # Only include user messages and Gemini's own responses
        if role == "user":
            gemini_history.append({"role": "user", "parts": [entry["content"]]})
        elif role == "gemini":
            gemini_history.append({"role": "model", "parts": [entry["content"]]})
        # Skip chatgpt, system, and other roles - they shouldn't be sent to Gemini
    return gemini_history


def format_history_for_chatgpt(history):
    """Convert history to ChatGPT API format

    Filters out responses from other LLMs (e.g., Gemini) to avoid
    sending ChatGPT messages it didn't generate, which would create
    a self-contradictory conversation.
    """
    chatgpt_history = []
    for entry in history:
        role = entry["role"]
        if role == "system":
            chatgpt_history.append({"role": "system", "content": entry["content"]})
        elif role == "user":
            chatgpt_history.append({"role": "user", "content": entry["content"]})
        elif role == "chatgpt":
            chatgpt_history.append({"role": "assistant", "content": entry["content"]})
        # Skip gemini and other roles - they shouldn't be sent to ChatGPT
    return chatgpt_history


def _estimate_tokens(text):
    """Estimate token count for mixed English/Japanese text

    More accurate estimation that accounts for Japanese characters:
    - ASCII/Latin: ~4 chars = 1 token
    - Japanese (hiragana/katakana/kanji): ~1.5 chars = 1 token

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Count Japanese characters (Unicode ranges for CJK)
    japanese_chars = sum(
        1
        for char in text
        if "\u3040" <= char <= "\u309f"  # Hiragana
        or "\u30a0" <= char <= "\u30ff"  # Katakana
        or "\u4e00" <= char <= "\u9fff"  # Kanji
        or "\uff00" <= char <= "\uffef"  # Full-width characters
    )

    ascii_chars = len(text) - japanese_chars

    # Japanese: 1.5 chars ≈ 1 token, ASCII: 4 chars ≈ 1 token
    estimated = (japanese_chars / 1.5) + (ascii_chars / 4.0)

    return int(estimated)


def get_token_info(text, model_name, history=None):
    """Get token information for the given text and model

    Args:
        text: System prompt text
        model_name: Model to use for context length calculation
        history: Optional conversation history to include in token count

    Returns:
        dict with token_count, max_context_length, and is_estimated
    """
    # Improved estimation with Japanese support
    estimated_tokens = _estimate_tokens(text)

    # Add history tokens if provided
    if history:
        history_text = "".join(entry.get("content", "") for entry in history)
        estimated_tokens += _estimate_tokens(history_text)

    # Define max context length per model (updated for GPT-4 variants and Gemini models)
    # Pattern matching is ordered from most specific to least specific
    model_lower = model_name.lower()

    # Model context patterns: (pattern, max_context)
    # Order matters: more specific patterns must come first
    MODEL_PATTERNS = [
        # Gemini models - specific variants first
        ("gemini-2.0-flash", 1048576),
        ("gemini-exp-1206", 1048576),
        ("gemini-1.5-pro", 2097152),
        ("gemini-1.5-flash", 1048576),
        ("gemini-pro", 32760),
        ("gemini", 32760),  # Conservative default for unknown Gemini
        # GPT models - specific variants first
        ("gpt-4o", 128000),
        ("gpt-4-turbo", 128000),
        ("gpt-4-1106", 128000),
        ("gpt-4", 8192),  # Base GPT-4
        ("gpt-3.5-turbo-16k", 16385),
        ("gpt-3.5", 4096),
    ]

    # Find first matching pattern
    max_context = 4096  # Conservative default
    for pattern, context_length in MODEL_PATTERNS:
        if pattern in model_lower:
            max_context = context_length
            break

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
        # Only add system message if prompt is not empty
        if system_prompt and system_prompt.strip():
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
        error_msg = f"Gemini API Error: An unexpected error occurred: {e}"
        print(error_msg)
        # Try to list available models (may also fail if auth/network issue)
        try:
            list_gemini_models()
        except Exception:
            pass  # Suppress secondary errors to avoid crash
        yield error_msg


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


def extract_text_from_chunk(chunk, model_name):
    """Extract text content from API response chunk

    Handles different response formats from Gemini and ChatGPT APIs.

    Args:
        chunk: Response chunk from LLM API
        model_name: Model identifier ("gemini" or "chatgpt")

    Returns:
        Extracted text string, or empty string if extraction fails
    """
    text = ""
    try:
        if model_name == "gemini":
            text = chunk.text
        elif model_name == "chatgpt":
            delta_content = chunk.choices[0].delta.content
            # Handle both string and list responses from OpenAI API
            if isinstance(delta_content, list):
                text = "".join(
                    part.text if hasattr(part, "text") else str(part) for part in delta_content
                )
            elif delta_content is not None:
                text = delta_content
    except (AttributeError, IndexError, TypeError, ValueError):
        # Fallback: treat chunk as string if extraction fails
        if isinstance(chunk, str):
            text = chunk

    return text
