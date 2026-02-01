import logging

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that depend on them
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# Import after load_dotenv() to ensure env vars are available  # noqa: E402

# Import legacy API wrappers from core_modules (DEPRECATED functions)
from .history_utils import (
    LLM_ROLES as LLM_ROLES,
)
from .llm_provider import (
    CHATGPT_MODEL as CHATGPT_MODEL,
)
from .llm_provider import (
    GEMINI_MODEL as GEMINI_MODEL,
)
from .llm_provider import (
    GOOGLE_API_KEY as GOOGLE_API_KEY,
)
from .llm_provider import (
    MCP_ENABLED as MCP_ENABLED,
)
from .llm_provider import (
    OPENAI_API_KEY as OPENAI_API_KEY,
)
from .llm_provider import (
    ChatGPTProvider as ChatGPTProvider,
)
from .llm_provider import (
    GeminiProvider as GeminiProvider,
)
from .llm_provider import (
    create_provider as create_provider,
)
from .llm_provider import (
    get_provider as get_provider,
)

# Import token and context management wrappers from core_modules

# Import Agentic Loop implementation from core_modules


# list_gemini_models remains here temporarily (will move to providers_facade later)
def list_gemini_models(verbose: bool = True):
    """List available Gemini models (debug utility)

    Args:
        verbose: If True, print models to stdout (default: True)

    Returns:
        list: List of available model names, or empty list if API key not configured
    """
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment variables or .env file.")
        return []

    genai.configure(api_key=GOOGLE_API_KEY)
    models = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            models.append(m.name)
            logger.debug("Available Gemini model: %s", m.name)

    if verbose:
        print("利用可能なGeminiモデル:")
        for name in models:
            print(f"  - {name}")

    logger.info("Found %d Gemini models", len(models))
    return models
