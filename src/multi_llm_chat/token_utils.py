import logging
import os

from dotenv import load_dotenv

load_dotenv()


def get_buffer_factor() -> float:
    """Get token estimation buffer factor from environment variable

    Returns:
        float: Buffer factor (default: 1.2)
    """
    try:
        return float(os.getenv("TOKEN_ESTIMATION_BUFFER_FACTOR", "1.2"))
    except ValueError as e:
        invalid_value = os.getenv("TOKEN_ESTIMATION_BUFFER_FACTOR")
        logging.warning(
            f"Invalid TOKEN_ESTIMATION_BUFFER_FACTOR: {invalid_value}. "
            f"Using default 1.2. Error: {e}"
        )
        return 1.2


def estimate_tokens(text: str) -> int:
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


def get_max_context_length(model_name: str) -> int:
    """Get maximum context length for the specified model

    Reads from environment variables with fallback to model defaults

    Args:
        model_name: Model identifier

    Returns:
        Maximum context length in tokens
    """
    model_lower = model_name.lower()

    # Check for model-specific environment variable
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

    # Fall back to default
    default_max = os.getenv("DEFAULT_MAX_CONTEXT_LENGTH")
    if default_max:
        try:
            return int(default_max)
        except ValueError:
            logging.warning(f"Invalid DEFAULT_MAX_CONTEXT_LENGTH: {default_max}. Using default.")

    # Built-in model-specific defaults
    MODEL_DEFAULTS = [
        ("gemini-2.0-flash", 1048576),
        ("gemini-exp-1206", 1048576),
        ("gemini-1.5-pro", 2097152),
        ("gemini-1.5-flash", 1048576),
        ("gemini-pro", 32760),
        ("gemini", 32760),
        ("gpt-4o", 128000),
        ("gpt-4-turbo", 128000),
        ("gpt-4-1106", 128000),
        ("gpt-4", 8192),
        ("gpt-3.5-turbo-16k", 16385),
        ("gpt-3.5", 4096),
    ]

    # Find first matching pattern
    for pattern, context_length in MODEL_DEFAULTS:
        if pattern in model_lower:
            return context_length

    return 4096
