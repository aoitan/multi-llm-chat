"""WebUI package - Gradio-based web interface"""

from ..history import HistoryStore  # For backward compatibility in tests
from .app import demo, launch
from .components import ASSISTANT_LABELS, ASSISTANT_ROLES, update_token_display
from .handlers import (
    check_history_name_exists,
    get_history_list,
    has_history_for_user,
    has_unsaved_session,
    load_history_action,
    new_chat_action,
    save_history_action,
    validate_and_respond,
)
from .state import WebUIState, hide_confirmation, show_confirmation

__all__ = [
    "launch",
    "demo",
    "update_token_display",
    "ASSISTANT_ROLES",
    "ASSISTANT_LABELS",
    "check_history_name_exists",
    "get_history_list",
    "has_history_for_user",
    "has_unsaved_session",
    "load_history_action",
    "new_chat_action",
    "validate_and_respond",
    "save_history_action",
    "WebUIState",
    "hide_confirmation",
    "show_confirmation",
    "HistoryStore",  # For backward compatibility
]
