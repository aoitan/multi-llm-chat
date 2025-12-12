import hashlib
import os
from collections import OrderedDict

import google.generativeai as genai
import openai
from dotenv import load_dotenv

try:
    import tiktoken  # noqa: F401

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo")
LLM_ROLES = {"gemini", "chatgpt"}

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

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use GeminiProvider.format_history() directly.
    """
    from multi_llm_chat.llm_provider import GeminiProvider

    return GeminiProvider.format_history(history)


def format_history_for_chatgpt(history):
    """Convert history to ChatGPT API format

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use ChatGPTProvider.format_history() directly.
    """
    from multi_llm_chat.llm_provider import ChatGPTProvider

    return ChatGPTProvider.format_history(history)


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

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.get_token_info() directly.

    Args:
        text: System prompt text
        model_name: Model to use for context length calculation
        history: Optional conversation history to include in token count

    Returns:
        dict with token_count, max_context_length, and is_estimated
    """
    from multi_llm_chat.llm_provider import get_provider

    # Determine provider from model name
    model_lower = model_name.lower()
    if "gpt" in model_lower or "chatgpt" in model_lower:
        provider_name = "chatgpt"
    else:
        provider_name = "gemini"

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


def prepare_request(history, system_prompt, model_name):
    """Prepare API request with system prompt and history"""
    if "gemini" in model_name.lower():
        # For Gemini, return tuple (system_prompt, history)
        # Only include system_prompt if it's not empty or whitespace-only
        if system_prompt and system_prompt.strip():
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
    """Call Gemini API with optional system prompt

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use llm_provider.get_provider("gemini") instead.
    """
    from multi_llm_chat.llm_provider import get_provider

    try:
        provider = get_provider("gemini")
        yield from provider.call_api(history, system_prompt)
    except ValueError as e:
        yield f"Gemini API Error: {e}"
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
    """Call ChatGPT API with optional system prompt

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use llm_provider.get_provider("chatgpt") instead.
    """
    from multi_llm_chat.llm_provider import get_provider

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

    Args:
        chunk: Response chunk from LLM API
        model_name: Model identifier ("gemini" or "chatgpt")

    Returns:
        Extracted text string, or empty string if extraction fails
    """
    from multi_llm_chat.llm_provider import get_provider

    try:
        provider_name = "gemini" if "gemini" in model_name.lower() else "chatgpt"
        provider = get_provider(provider_name)
        return provider.extract_text_from_chunk(chunk)
    except Exception:
        # Fallback: treat chunk as string if extraction fails
        if isinstance(chunk, str):
            return chunk
        return ""


# Context compression and token guard rail functions


def get_max_context_length(model_name):
    """Get maximum context length for the specified model

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use llm_provider._get_max_context_length() directly.

    Args:
        model_name: Model identifier

    Returns:
        Maximum context length in tokens
    """
    from multi_llm_chat.llm_provider import _get_max_context_length

    return _get_max_context_length(model_name)


def calculate_tokens(text, model_name):
    """Calculate token count for text using model-appropriate method

    DEPRECATED: This is a backward compatibility wrapper.
    New code should use provider.get_token_info() directly.

    Args:
        text: Text to tokenize
        model_name: Model identifier

    Returns:
        Token count (int)
    """
    from multi_llm_chat.llm_provider import get_provider

    # Determine provider from model name
    model_lower = model_name.lower()
    if "gpt" in model_lower or "chatgpt" in model_lower:
        provider_name = "chatgpt"
    else:
        provider_name = "gemini"

    # Get provider and calculate tokens for single message with actual model name
    provider = get_provider(provider_name)
    result = provider.get_token_info(text, history=None, model_name=model_name)
    return result["input_tokens"]


def prune_history_sliding_window(history, max_tokens, model_name, system_prompt=None):
    """Prune conversation history using sliding window approach

    Removes oldest conversation turns to fit within max_tokens limit.
    Always preserves the most recent turns and accounts for system prompt.
    Preserves user-assistant turn pairs to maintain conversation coherence.

    Args:
        history: Conversation history list
        max_tokens: Maximum tokens allowed
        model_name: Model identifier
        system_prompt: Optional system prompt (not included in history)

    Returns:
        Pruned history list
    """
    if not history:
        return history

    # Calculate system prompt tokens
    system_tokens = 0
    if system_prompt:
        system_tokens = calculate_tokens(system_prompt, model_name)

    # Calculate tokens for each entry
    entry_tokens = []
    for entry in history:
        content = entry.get("content", "")
        tokens = calculate_tokens(content, model_name)
        entry_tokens.append(tokens)

    # Calculate total tokens
    total_tokens = system_tokens + sum(entry_tokens)

    # If within limit, return as-is
    if total_tokens <= max_tokens:
        return history

    # Prune from the beginning, preserving complete user-assistant turns
    # A turn can have multiple assistant responses (@all pattern)
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


def validate_system_prompt_length(system_prompt, model_name):
    """Validate that system prompt doesn't exceed model's max context

    Args:
        system_prompt: System prompt text
        model_name: Model identifier

    Returns:
        dict with 'valid' (bool) and optional 'error' (str)
    """
    if not system_prompt:
        return {"valid": True}

    max_length = get_max_context_length(model_name)
    prompt_tokens = calculate_tokens(system_prompt, model_name)

    if prompt_tokens > max_length:
        return {
            "valid": False,
            "error": (
                f"System prompt ({prompt_tokens} tokens) exceeds "
                f"max context length ({max_length} tokens)"
            ),
        }

    return {"valid": True}


def validate_context_length(history, system_prompt, model_name):
    """Validate that system prompt + latest turn doesn't exceed max context

    Args:
        history: Conversation history
        system_prompt: System prompt text
        model_name: Model identifier

    Returns:
        dict with 'valid' (bool) and optional 'error' (str)
    """
    max_length = get_max_context_length(model_name)

    # Calculate system prompt tokens
    system_tokens = 0
    if system_prompt:
        system_tokens = calculate_tokens(system_prompt, model_name)

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
            latest_tokens += calculate_tokens(content, model_name)
            i -= 1

        # Add user message if found
        if i >= 0 and history[i]["role"] == "user":
            content = history[i].get("content", "")
            latest_tokens += calculate_tokens(content, model_name)

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


def get_pruning_info(history, max_tokens, model_name, system_prompt=None):
    """Get information about how history would be pruned

    Args:
        history: Conversation history
        max_tokens: Maximum tokens allowed
        model_name: Model identifier
        system_prompt: Optional system prompt

    Returns:
        dict with pruning statistics
    """
    if not history:
        return {
            "turns_to_remove": 0,
            "original_length": 0,
            "pruned_length": 0,
        }

    # Calculate current total
    system_tokens = 0
    if system_prompt:
        system_tokens = calculate_tokens(system_prompt, model_name)

    original_tokens = system_tokens
    for entry in history:
        content = entry.get("content", "")
        original_tokens += calculate_tokens(content, model_name)

    # Get pruned version
    pruned = prune_history_sliding_window(history, max_tokens, model_name, system_prompt)

    pruned_tokens = system_tokens
    for entry in pruned:
        content = entry.get("content", "")
        pruned_tokens += calculate_tokens(content, model_name)

    turns_removed = len(history) - len(pruned)

    return {
        "turns_to_remove": turns_removed,
        "original_length": original_tokens,
        "pruned_length": pruned_tokens,
    }
