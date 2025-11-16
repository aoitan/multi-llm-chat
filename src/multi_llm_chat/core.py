import hashlib
import logging
import os
from collections import OrderedDict

import google.generativeai as genai
import openai
from dotenv import load_dotenv

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

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
    # Use calculate_tokens for accurate/buffered counting
    token_count = calculate_tokens(text, model_name)

    # Determine model type for filtering
    model_lower = model_name.lower()

    # Add history tokens if provided (filter by model)
    if history:
        # Determine which roles to count based on model
        if "gpt" in model_lower:
            # For ChatGPT: count user and chatgpt messages only
            relevant_roles = {"user", "chatgpt"}
        else:
            # For Gemini: count user and gemini messages only
            relevant_roles = {"user", "gemini"}

        for entry in history:
            role = entry.get("role", "")
            if role in relevant_roles:
                content = entry.get("content", "")
                token_count += calculate_tokens(content, model_name)

    # Use environment-based max context length (from get_max_context_length)
    # This ensures consistency with context compression logic
    max_context = get_max_context_length(model_name)

    # Check if using tiktoken (accurate) or estimation
    is_estimated = "gpt" not in model_lower or not TIKTOKEN_AVAILABLE

    return {
        "token_count": token_count,
        "max_context_length": max_context,
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


# Context compression and token guard rail functions


def get_max_context_length(model_name):
    """Get maximum context length for the specified model

    Reads from environment variables with fallback to model defaults:
    1. Model-specific: GEMINI_MAX_CONTEXT_LENGTH, CHATGPT_MAX_CONTEXT_LENGTH
    2. Generic: DEFAULT_MAX_CONTEXT_LENGTH
    3. Model built-in defaults (based on model capabilities)

    Args:
        model_name: Model identifier

    Returns:
        Maximum context length in tokens
    """
    model_lower = model_name.lower()

    # Check for model-specific environment variable (read dynamically)
    if "gemini" in model_lower:
        gemini_max = os.getenv("GEMINI_MAX_CONTEXT_LENGTH")
        if gemini_max:
            try:
                return int(gemini_max)
            except ValueError:
                logging.warning(f"Invalid GEMINI_MAX_CONTEXT_LENGTH: {gemini_max}. Using default.")

    if "gpt" in model_lower:
        chatgpt_max = os.getenv("CHATGPT_MAX_CONTEXT_LENGTH")
        if chatgpt_max:
            try:
                return int(chatgpt_max)
            except ValueError:
                logging.warning(
                    f"Invalid CHATGPT_MAX_CONTEXT_LENGTH: {chatgpt_max}. Using default."
                )

    # Fall back to default (also read dynamically)
    default_max = os.getenv("DEFAULT_MAX_CONTEXT_LENGTH")
    if default_max:
        try:
            return int(default_max)
        except ValueError:
            logging.warning(f"Invalid DEFAULT_MAX_CONTEXT_LENGTH: {default_max}. Using default.")

    # Built-in model-specific defaults (based on model capabilities)
    # Pattern matching is ordered from most specific to least specific
    MODEL_DEFAULTS = [
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
    for pattern, context_length in MODEL_DEFAULTS:
        if pattern in model_lower:
            return context_length

    # Final fallback
    return 4096


def calculate_tokens(text, model_name):
    """Calculate token count for text using model-appropriate method

    - OpenAI models: Use tiktoken for accurate counting (includes message overhead)
    - Other models: Use estimation with buffer factor

    Args:
        text: Text to tokenize
        model_name: Model identifier

    Returns:
        Token count (int)
    """
    model_lower = model_name.lower()

    # Use tiktoken for OpenAI models if available
    if "gpt" in model_lower and TIKTOKEN_AVAILABLE:
        try:
            # Map model name to tiktoken encoding
            if "gpt-4" in model_lower:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model_lower:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Fallback to cl100k_base (used by gpt-4, gpt-3.5-turbo)
                encoding = tiktoken.get_encoding("cl100k_base")

            content_tokens = len(encoding.encode(text))

            # Add per-message overhead for Chat Completions format
            # OpenAI guidance: ~3 tokens per message for role, separators, etc.
            # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            message_overhead = 3

            return content_tokens + message_overhead
        except Exception:
            # Fall back to estimation if tiktoken fails
            pass

    # Use estimation with buffer factor for other models
    try:
        buffer_factor = float(os.getenv("TOKEN_ESTIMATION_BUFFER_FACTOR", "1.2"))
    except ValueError as e:
        invalid_value = os.getenv("TOKEN_ESTIMATION_BUFFER_FACTOR")
        logging.warning(
            f"Invalid TOKEN_ESTIMATION_BUFFER_FACTOR: {invalid_value}. "
            f"Using default 1.2. Error: {e}"
        )
        buffer_factor = 1.2

    base_estimate = _estimate_tokens(text)
    return int(base_estimate * buffer_factor)


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
