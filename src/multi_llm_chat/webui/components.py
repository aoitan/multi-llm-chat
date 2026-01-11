"""UI components and utilities for WebUI"""

import logging

from gradio_client import utils as gradio_utils

from .. import core

logger = logging.getLogger(__name__)

# Constants
ASSISTANT_ROLES = ("assistant", "gemini", "chatgpt")
ASSISTANT_LABELS = {
    "assistant": "**Assistant:**\n",
    "gemini": "**Gemini:**\n",
    "chatgpt": "**ChatGPT:**\n",
}

# Gradio 4.42.0時点のバグで、JSON Schema内にboolが含まれると
# gradio_client.utils.json_schema_to_python_typeが落ちる。
# Blocks.launch()時にAPI情報の生成で呼ばれるため、
# ここでboolを扱えるように安全なラッパーを差し込む。
_orig_json_schema_to_python_type = gradio_utils.json_schema_to_python_type
_orig__json_schema_to_python_type = gradio_utils._json_schema_to_python_type
_orig_get_type = gradio_utils.get_type


def _safe_json_schema_to_python_type(schema):  # pragma: no cover - runtime patch
    if isinstance(schema, bool):
        return "bool" if schema else "Never"
    return _orig_json_schema_to_python_type(schema)


def _safe__json_schema_to_python_type(schema, defs):  # pragma: no cover - runtime patch
    if isinstance(schema, bool):
        return "bool" if schema else "Never"
    return _orig__json_schema_to_python_type(schema, defs)


def _safe_get_type(schema):  # pragma: no cover - runtime patch
    if isinstance(schema, bool):
        return "boolean" if schema else "const"
    return _orig_get_type(schema)


# Apply patches
gradio_utils.json_schema_to_python_type = _safe_json_schema_to_python_type
gradio_utils._json_schema_to_python_type = _safe__json_schema_to_python_type
gradio_utils.get_type = _safe_get_type


def update_token_display(system_prompt, logic_history=None, model_name=None):
    """Update token count display for system prompt and conversation history

    Args:
        system_prompt: System prompt text
        logic_history: Current conversation history (optional)
        model_name: Model name for context length calculation (optional)

    Returns:
        HTML string for token display
    """
    if model_name is None:
        # Use smallest context length to be conservative
        model_name = core.CHATGPT_MODEL

    if not system_prompt:
        return "Tokens: 0 / - (no system prompt)"

    # Include history in token calculation
    token_info = core.get_token_info(system_prompt, model_name, logic_history)
    token_count = token_info["token_count"]
    max_context = token_info["max_context_length"]
    is_estimated = token_info["is_estimated"]

    estimation_note = " (estimated)" if is_estimated else ""

    if token_count > max_context:
        return (
            f'<span style="color: red;">警告: Tokens: {token_count} / {max_context}'
            f"{estimation_note} - 上限を超えています</span>"
        )
    else:
        return f"Tokens: {token_count} / {max_context}{estimation_note}"
