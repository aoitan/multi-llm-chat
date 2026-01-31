import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

LLM_ROLES = {"gemini", "chatgpt"}
TOOL_ROLES = {"tool"}
ALL_ROLES = LLM_ROLES | {"user"} | TOOL_ROLES
logger = logging.getLogger(__name__)


def validate_history_entry(entry: Dict[str, Any]) -> None:
    """Validate a single history entry for structural correctness.

    Args:
        entry: History entry to validate

    Raises:
        ValueError: If entry structure is invalid

    Examples:
        >>> validate_history_entry({"role": "user", "content": "hello"})
        # Valid - no exception

        >>> validate_history_entry({"role": "tool", "content": [
        ...     {"type": "tool_result", "tool_call_id": "call_123", "content": "OK"}
        ... ]})
        # Valid - no exception

        >>> validate_history_entry({"role": "invalid_role", "content": "test"})
        # Raises ValueError
    """
    if not isinstance(entry, dict):
        raise ValueError(f"History entry must be dict, got {type(entry).__name__}")

    role = entry.get("role")
    if role not in ALL_ROLES:
        raise ValueError(f"Invalid role: '{role}'. Must be one of {ALL_ROLES}")

    content = entry.get("content")
    if content is None:
        raise ValueError("History entry must have 'content' field")

    # String content is legacy but still allowed (will be deprecated in v2.0.0)
    if isinstance(content, str):
        return

    # Structured content must be a list
    if not isinstance(content, list):
        raise ValueError(f"Structured content must be list, got {type(content).__name__}")

    # Special validation for role: "tool"
    if role == "tool":
        if not content:
            raise ValueError("role='tool' must have non-empty content")
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                raise ValueError(
                    f"role='tool' content[{i}] must be dict, got {type(item).__name__}"
                )
            item_type = item.get("type")
            if item_type != "tool_result":
                raise ValueError(
                    f"role='tool' can only contain type='tool_result', "
                    f"got type='{item_type}' at content[{i}]"
                )
            if not item.get("tool_call_id"):
                raise ValueError(f"tool_result must have 'tool_call_id' field at content[{i}]")


def history_contains_tools(history: List[Dict[str, Any]]) -> bool:
    """Check if history contains any tool calls or tool results.

    Args:
        history: Conversation history

    Returns:
        True if history contains tool-related content
    """
    for entry in history:
        content = entry.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("tool_call", "tool_result"):
                    return True
    return False


def _stringify_tool_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(payload)


def content_to_text(content: Any, include_tool_data: bool = False) -> str:
    """Normalize history content into text for token calculations.

    This function converts structured history content (text, tool_call, tool_result)
    into a plain text representation suitable for token counting.

    DEPRECATION NOTICE (v2.0.0+):
        String-based content will trigger a DeprecationWarning in v2.0.0.
        Use structured content format (List[Dict]) for all new code.
        See doc/migration_plan.md for details.

    Important Note on Token Estimation Accuracy:
        When include_tool_data=True, tool_call and tool_result are serialized
        to JSON strings for token estimation. This is an APPROXIMATION and may
        underestimate tokens by 10-30% for complex tool schemas due to:
        - Gemini's FunctionDeclaration includes schema metadata (type, description)
        - Tool result wrapping (function_response structure) adds overhead

        The BUFFER_FACTOR (default: 1.3) in token_utils.py compensates for this,
        but for tool-heavy workflows, consider increasing it to 1.5:
            export TOKEN_ESTIMATION_BUFFER_FACTOR=1.5

        Future improvement: Integrate with Gemini's actual token counting API
        when available (tracked in migration_plan.md).

    Args:
        content: History content (str, list of dicts with "type" and "content", or None)
        include_tool_data: If True, include tool calls and results in calculation
            (used by compression, validation, and token info)

    Returns:
        Concatenated text string for token counting. Empty string if content is None.

    Examples:
        >>> content_to_text("Hello")
        "Hello"
        >>> content_to_text([{"type": "text", "content": "Hi"}])
        "Hi"
        >>> content_to_text(
        ...     [{"type": "text", "content": "Hi"},
        ...      {"type": "tool_call", "content": {...}}],
        ...     include_tool_data=True
        ... )
        "Hi search {\"query\": \"python\"}"
    """
    if isinstance(content, list):
        text_parts = []
        tool_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text_parts.append(part.get("content", ""))
            elif include_tool_data and part_type == "tool_call":
                tool_call = part.get("content", {}) if isinstance(part.get("content"), dict) else {}
                name = tool_call.get("name")
                args = _stringify_tool_payload(tool_call.get("arguments"))
                tool_parts.append(" ".join(item for item in [name, args] if item))
            elif include_tool_data and part_type == "tool_result":
                name = part.get("name")
                result = _stringify_tool_payload(part.get("content"))
                tool_parts.append(" ".join(item for item in [name, result] if item))

        combined = [part for part in text_parts + tool_parts if part]
        return " ".join(combined)
    if content is None:
        return ""
    # Legacy string format - will be deprecated in v2.0.0
    if isinstance(content, str):
        # Note: Warning will be enabled in Phase 2 (v2.0.0)
        # warnings.warn(
        #     "String-based content is deprecated. Use List[Dict] format. "
        #     "See doc/migration_plan.md for migration guide.",
        #     DeprecationWarning,
        #     stacklevel=2
        # )
        return content
    return str(content)


def normalize_history_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize legacy history entries into structured content lists.

    Invalid entries (non-dict types) are replaced with placeholder entries
    to preserve index alignment for functions like get_llm_response().
    """
    normalized = []
    for original_index, entry in enumerate(turns or []):
        if not isinstance(entry, dict):
            logger.warning(
                "Invalid history entry at index %s: expected dict, got %s. "
                "Replacing with placeholder.",
                original_index,
                type(entry).__name__,
            )
            # Replace with placeholder to preserve index alignment
            normalized.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "content": "[Invalid entry removed]"}],
                    "invalid": True,  # Flag for future filtering
                }
            )
            continue
        role = entry.get("role")
        content = entry.get("content")
        normalized_entry = entry.copy()
        normalized_entry["content"] = _normalize_content_parts(content, role, entry.get("name"))
        normalized.append(normalized_entry)
    return normalized


def _normalize_content_parts(
    content: Any, role: Optional[str], tool_name: Optional[str]
) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part)
            elif isinstance(part, str):
                parts.append({"type": "text", "content": part})
        return parts
    if content is None:
        return []
    if isinstance(content, str):
        if role == "tool":
            return [{"type": "tool_result", "name": tool_name, "content": content}]
        return [{"type": "text", "content": content}]
    return [{"type": "text", "content": str(content)}]


def get_provider_name_from_model(model_name: str) -> str:
    """Get provider name from model name

    Args:
        model_name: Model identifier

    Returns:
        str: Provider name ("gemini" or "chatgpt")
    """
    model_lower = model_name.lower()
    if "gpt" in model_lower or "chatgpt" in model_lower:
        return "chatgpt"
    return "gemini"


def prepare_request(
    history: List[Dict[str, Any]], system_prompt: str, model_name: str
) -> Union[List[Dict[str, Any]], Tuple[Optional[str], List[Dict[str, Any]]]]:
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


# ========================================
# Text extraction and chunk processing
# (Added for Issue #103 refactoring)
# ========================================
# Functions for extracting text from streaming chunks
# will be added here during Phase 2 migration
