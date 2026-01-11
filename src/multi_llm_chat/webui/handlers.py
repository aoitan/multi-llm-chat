"""Event handlers for WebUI"""

import logging

from ..chat_logic import ChatService
from ..history import HistoryStore
from .components import ASSISTANT_LABELS, ASSISTANT_ROLES

logger = logging.getLogger(__name__)


def logic_history_to_display(logic_history):
    """Converts logic history to display history format."""
    display_history = []
    for turn in logic_history:
        if turn["role"] == "user":
            display_history.append([turn["content"], ""])
        elif turn["role"] in ASSISTANT_ROLES and display_history:
            prefix = ASSISTANT_LABELS.get(turn["role"], "")
            formatted_content = f"{prefix}{turn['content']}" if prefix else turn["content"]
            current_response = display_history[-1][1]
            if current_response:
                display_history[-1][1] = current_response + "\n\n" + formatted_content
            else:
                display_history[-1][1] = formatted_content
    return display_history


def save_history_action(user_id, save_name, logic_history, system_prompt):
    """Save current chat history using HistoryStore

    Args:
        user_id: User ID
        save_name: Name to save the history as
        logic_history: Logic history
        system_prompt: System prompt

    Returns:
        tuple: (status_message, updated_dropdown_choices)
    """
    try:
        store = HistoryStore()
        store.save_history(user_id, save_name, system_prompt, logic_history)

        # Get updated list of histories
        choices = store.list_histories(user_id)

        return (f"✅ 履歴 '{save_name}' を保存しました", choices)
    except ValueError as e:
        logger.error(f"Error saving history '{save_name}' for user '{user_id}': {e}", exc_info=True)
        return (f"❌ 保存エラー: {e}", [])
    except Exception as e:
        logger.error(
            f"Unexpected error saving history '{save_name}' for user '{user_id}': {e}",
            exc_info=True,
        )
        return (f"❌ 保存に失敗しました: {e}", [])


def load_history_action(user_id, history_name):
    """Load saved chat history using HistoryStore

    Args:
        user_id: User ID
        history_name: Name of history to load

    Returns:
        tuple: (display_history, logic_history, system_prompt, status_message)
    """
    try:
        store = HistoryStore()
        data = store.load_history(user_id, history_name)

        system_prompt = data.get("system_prompt", "")
        logic_history = data.get("turns", [])

        # Convert logic history to display history
        display_history = []
        for turn in logic_history:
            if turn["role"] == "user":
                # Start a new turn
                display_history.append([turn["content"], ""])
            elif turn["role"] in ASSISTANT_ROLES and display_history:
                # Add assistant/LLM response to the last turn
                # For @all mentions, multiple assistant responses exist - append them
                prefix = ASSISTANT_LABELS.get(turn["role"], "")
                formatted_content = f"{prefix}{turn['content']}" if prefix else turn["content"]
                current_response = display_history[-1][1]
                if current_response:
                    # Already has a response, append the new one
                    display_history[-1][1] = current_response + "\n\n" + formatted_content
                else:
                    # First response for this user message
                    display_history[-1][1] = formatted_content

        return (
            display_history,
            logic_history,
            system_prompt,
            f"✅ 履歴 '{history_name}' を読み込みました",
        )
    except FileNotFoundError:
        # Return None to indicate error - caller should preserve current state
        return (None, None, None, f"❌ 履歴 '{history_name}' が見つかりません")
    except Exception as e:
        # Return None to indicate error - caller should preserve current state
        return (None, None, None, f"❌ 読み込みに失敗しました: {e}")


def new_chat_action():
    """Start new chat session

    Returns:
        tuple: (display_history, logic_history, system_prompt, status_message)
    """
    return ([], [], "", "✅ 新しい会話を開始しました")


def has_unsaved_session(logic_history):
    """Check if there's an unsaved session

    Args:
        logic_history: Current logic history

    Returns:
        bool: True if there's unsaved content
    """
    # Session is unsaved if there's any conversation history
    return len(logic_history) > 0


def check_history_name_exists(user_id, save_name):
    """Check if a history name already exists using HistoryStore

    Args:
        user_id: User ID
        save_name: Name to check

    Returns:
        bool: True if name exists
    """
    try:
        store = HistoryStore()
        return store.history_exists(user_id, save_name)
    except Exception as e:
        logger.error(f"Failed to check history existence for user '{user_id}': {e}")
        return False


def get_history_list(user_id):
    """Get list of saved histories for user

    Args:
        user_id: User ID

    Returns:
        list: List of history names (empty list on error)
    """
    if not user_id or not user_id.strip():
        return []

    try:
        store = HistoryStore()
        return store.list_histories(user_id)
    except Exception as e:
        logger.error(f"Failed to list histories for user '{user_id}': {e}")
        return []


def has_history_for_user(user_id: str) -> bool:
    """Check if a user has any saved histories."""
    if not user_id or not user_id.strip():
        return False
    return len(get_history_list(user_id)) > 0


def respond(user_message, display_history, logic_history, system_prompt, user_id, chat_service):
    """
    検証済みの入力に基づき、チャットの中核的な応答ロジック（LLM呼び出し、履歴管理）を実行します。

    この関数は入力検証を行いません。呼び出し元は、この関数を呼び出す前に
    user_id が有効であることを保証する必要があります。
    UI統合には、検証をラップした `validate_and_respond()` を使用してください。

    Args:
        user_message: ユーザーの入力メッセージ
        display_history: チャットボットUIに表示するための履歴
        logic_history: 内部で管理するための論理履歴
        system_prompt: システムプロンプト
        user_id: ユーザーID（呼び出し元で検証済みであること）
        chat_service: セッションスコープのChatServiceインスタンス (gr.Stateより)

    Yields:
        tuple: (display_history, display_history, logic_history, chat_service)
    """
    # Initialize or reuse ChatService (session-scoped for provider reuse)
    # Reset service if history was cleared (e.g., new chat action)
    if chat_service is None or not logic_history:
        chat_service = ChatService()

    # Update service state with current histories and system prompt
    chat_service.display_history = display_history
    chat_service.logic_history = logic_history
    chat_service.system_prompt = system_prompt

    for updated_display, updated_logic in chat_service.process_message(user_message):
        yield updated_display, updated_display, updated_logic, chat_service


def validate_and_respond(
    user_message, display_history, logic_history, system_prompt, user_id, chat_service
):
    """
    入力検証と応答処理をラップするジェネレータ。
    user_idを検証し、無効な場合はエラーを返し、有効な場合は `respond` に処理を委譲します。
    """
    if not user_id or not user_id.strip():
        display_history.append([user_message, "[System: ユーザーIDを入力してください]"])
        yield display_history, display_history, logic_history, chat_service
        return

    yield from respond(
        user_message, display_history, logic_history, system_prompt, user_id, chat_service
    )
