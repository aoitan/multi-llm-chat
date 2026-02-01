import logging

from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that depend on them
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# Import after load_dotenv() to ensure env vars are available  # noqa: E402

# Import legacy API wrappers from core_modules (DEPRECATED functions)
# Import Agentic Loop implementation from core_modules
from .core_modules.agentic_loop import (  # noqa: F401
    AgenticLoopResult,
    execute_with_tools,
    execute_with_tools_stream,
    execute_with_tools_sync,
)
from .core_modules.legacy_api import (  # noqa: F401
    call_chatgpt_api,
    call_chatgpt_api_async,
    call_gemini_api,
    call_gemini_api_async,
    extract_text_from_chunk,
    format_history_for_chatgpt,
    format_history_for_gemini,
    load_api_key,
    prepare_request,
    stream_text_events,
    stream_text_events_async,
)

# Import provider facade from core_modules
from .core_modules.providers_facade import list_gemini_models  # noqa: F401

# Import token and context management wrappers from core_modules
from .core_modules.token_and_context import (  # noqa: F401
    calculate_tokens,
    get_max_context_length,
    get_pruning_info,
    get_token_info,
    prune_history_sliding_window,
    validate_context_length,
    validate_system_prompt_length,
)
from .history_utils import LLM_ROLES  # noqa: F401
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
