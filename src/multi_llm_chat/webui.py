import logging
import os

import gradio as gr
from gradio_client import utils as gradio_utils

from . import core
from .history import HistoryStore

# Constants
ASSISTANT_ROLES = ("assistant", "gemini", "chatgpt")

logger = logging.getLogger(__name__)

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
                current_response = display_history[-1][1]
                if current_response:
                    # Already has a response, append the new one
                    display_history[-1][1] = current_response + "\n\n" + turn["content"]
                else:
                    # First response for this user message
                    display_history[-1][1] = turn["content"]

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


def _build_history_operation_updates(user_id, display_hist, logic_hist, sys_prompt, status):
    """Build consistent UI update dictionary after history operations

    This helper consolidates common UI update logic used across multiple event handlers
    (load history, new chat, save history) to reduce code duplication.

    Args:
        user_id: User ID
        display_hist: Display history
        logic_hist: Logic history
        sys_prompt: System prompt
        status: Status message to display

    Returns:
        dict: Dictionary with keys matching Gradio output components
    """
    # Update token display and send button state
    token_display_value = update_token_display(sys_prompt, logic_hist)
    send_button_state = check_send_button_with_user_id(user_id, sys_prompt, logic_hist)

    # Get updated history list
    history_choices = get_history_list(user_id)

    return {
        "chatbot_ui": display_hist,
        "display_history_state": display_hist,
        "logic_history_state": logic_hist,
        "system_prompt_input": sys_prompt,
        "history_status": status,
        "token_display": token_display_value,
        "send_button": send_button_state,
        "history_dropdown": gr.update(choices=history_choices),
    }


def respond(user_message, display_history, logic_history, system_prompt, user_id):
    """
    ユーザー入力への応答、LLM呼び出し、履歴管理をすべて行う単一の関数。

    Args:
        user_message: User's input message
        display_history: Display history for chatbot UI
        logic_history: Internal logic history
        system_prompt: System prompt text
        user_id: User ID (required - must not be empty)
    """
    # Validate user_id before processing
    if not user_id or not user_id.strip():
        # Return error message without calling LLM
        display_history.append([user_message, "[System: ユーザーIDを入力してください]"])
        yield display_history, display_history, logic_history
        return

    def _stream_response(model_name, stream):
        full_response = ""
        for chunk in stream:
            text = core.extract_text_from_chunk(chunk, model_name)
            if text:
                full_response += text
                display_history[-1][1] += text
                yield display_history, display_history, logic_history
        return full_response

    # 1. ユーザーメッセージを両方の履歴に追加
    logic_history.append({"role": "user", "content": user_message})
    display_history.append([user_message, None])
    yield display_history, display_history, logic_history

    # 2. メンションを解析
    mention = ""
    msg_stripped = user_message.strip()
    if msg_stripped.startswith("@gemini"):
        mention = "gemini"
    elif msg_stripped.startswith("@chatgpt"):
        mention = "chatgpt"
    elif msg_stripped.startswith("@all"):
        mention = "all"

    history_snapshot = [entry.copy() for entry in logic_history] if mention == "all" else None

    # 3. Geminiへの応答処理
    if mention in ["gemini", "all"]:
        display_history[-1][1] = "**Gemini:**\n"
        gemini_input_history = history_snapshot or logic_history
        gemini_stream = core.call_gemini_api(gemini_input_history, system_prompt)
        full_response_g = yield from _stream_response("gemini", gemini_stream)

        logic_history.append({"role": "gemini", "content": full_response_g})
        if not full_response_g.strip():
            display_history[-1][1] = "**Gemini:**\n[System: Geminiからの応答がありませんでした]"
        yield display_history, display_history, logic_history

    # 4. ChatGPTへの応答処理
    if mention in ["chatgpt", "all"]:
        # @all の場合、UIの重複を避けるため、プロンプトなしで新しい行を追加
        if mention == "all":
            display_history.append([None, "**ChatGPT:**\n"])
        else:
            display_history[-1][1] = "**ChatGPT:**\n"

        chatgpt_input_history = history_snapshot or logic_history
        chatgpt_stream = core.call_chatgpt_api(chatgpt_input_history, system_prompt)
        full_response_c = yield from _stream_response("chatgpt", chatgpt_stream)

        logic_history.append({"role": "chatgpt", "content": full_response_c})
        if not full_response_c.strip():
            display_history[-1][1] = "**ChatGPT:**\n[System: ChatGPTからの応答がありませんでした]"
        yield display_history, display_history, logic_history


def check_history_buttons_enabled(user_id):
    """Check if history buttons should be enabled based on user_id

    Args:
        user_id: User ID string

    Returns:
        dict with button states using gr.update()
    """
    enabled = bool(user_id and user_id.strip())
    return {
        "save_btn": gr.update(interactive=enabled),
        "load_btn": gr.update(interactive=enabled),
        "new_btn": gr.update(interactive=enabled),
    }


def check_send_button_with_user_id(user_id, system_prompt, logic_history=None, model_name=None):
    """Check if send button should be enabled based on user_id AND token limit

    Args:
        user_id: User ID string (must not be empty)
        system_prompt: System prompt text
        logic_history: Current conversation history (optional)
        model_name: Model name for context length calculation (optional)

    Returns:
        gr.update() with interactive state
    """
    # First check user_id
    if not user_id or not user_id.strip():
        return gr.update(interactive=False)

    # Then check token limit
    if model_name is None:
        model_name = core.CHATGPT_MODEL

    if not system_prompt:
        return gr.update(interactive=True)

    token_info = core.get_token_info(system_prompt, model_name, logic_history)
    is_enabled = token_info["token_count"] <= token_info["max_context_length"]
    return gr.update(interactive=is_enabled)


def show_confirmation(message, action, data=None):
    """Show confirmation dialog with message

    Args:
        message: Confirmation message to display
        action: Action identifier (e.g., "save_overwrite", "load_unsaved")
        data: Optional data to pass to the action

    Returns:
        tuple: (dialog_visibility, message_content, state)
    """
    return (
        gr.update(visible=True),  # confirmation_dialog
        message,  # confirmation_message
        {"pending_action": action, "pending_data": data},  # confirmation_state
    )


def hide_confirmation():
    """Hide confirmation dialog and clear state

    Returns:
        tuple: (dialog_visibility, message_content, state)
    """
    return (
        gr.update(visible=False),  # confirmation_dialog
        "",  # confirmation_message
        {"pending_action": None, "pending_data": None},  # confirmation_state
    )


# --- Gradio UIの構築 ---
with gr.Blocks() as demo:
    gr.Markdown("# Multi-LLM Chat")

    # User ID input with warning
    with gr.Row():
        user_id_input = gr.Textbox(
            label="User ID",
            placeholder="Enter your user ID...",
            elem_id="user_id_input",
        )
    gr.Markdown(
        "⚠️ **注意**: これは認証ではありません。他人のIDを使わないでください。",
        elem_id="user_id_warning",
    )

    # System prompt input
    with gr.Row():
        system_prompt_input = gr.Textbox(
            label="System Prompt",
            placeholder="Enter system prompt (optional)...",
            lines=3,
        )

    # Token count display
    token_display = gr.Markdown("Tokens: 0 / - (no system prompt)")

    # History management panel
    with gr.Accordion("履歴管理", open=False):
        with gr.Row():
            history_dropdown = gr.Dropdown(
                label="保存済み履歴",
                choices=[],
                elem_id="history_dropdown",
            )
        with gr.Row():
            save_name_input = gr.Textbox(
                label="保存名",
                placeholder="履歴の名前を入力...",
                elem_id="save_name_input",
            )
        with gr.Row():
            save_history_btn = gr.Button(
                "現在の会話を保存", elem_id="save_history_btn", interactive=False
            )
            load_history_btn = gr.Button(
                "選択した会話を読み込む", elem_id="load_history_btn", interactive=False
            )
            new_chat_btn = gr.Button("新しい会話を開始", elem_id="new_chat_btn", interactive=False)
        history_status = gr.Markdown("", elem_id="history_status")

    # Confirmation dialog (modal-like UI)
    with gr.Row(visible=False, elem_id="confirmation_dialog") as confirmation_dialog:
        with gr.Column():
            confirmation_message = gr.Markdown("", elem_id="confirmation_message")
            with gr.Row():
                confirmation_yes_btn = gr.Button("Yes", elem_id="confirmation_yes_btn", size="sm")
                confirmation_no_btn = gr.Button("No", elem_id="confirmation_no_btn", size="sm")

    # Confirmation state management
    confirmation_state = gr.State(
        {"pending_action": None, "pending_data": None}
    )  # Stores pending action and data

    # 履歴を管理するための非表示Stateコンポーネント
    display_history_state = gr.State([])
    logic_history_state = gr.State([])

    # UIコンポーネント
    chatbot_ui = gr.Chatbot(label="Conversation", height=600, show_copy_button=True)

    with gr.Row():
        user_input = gr.Textbox(
            show_label=False,
            placeholder="Enter text with @mention...",
            container=False,
            scale=4,
        )
        send_button = gr.Button("Send", variant="primary", scale=1, interactive=False)
        reset_button = gr.Button(
            "新しい会話を開始", variant="secondary", scale=1, interactive=False
        )

    # Update button states when user ID changes
    def update_buttons_on_user_id(user_id, system_prompt, logic_history):
        enabled = bool(user_id and user_id.strip())
        history_choices = get_history_list(user_id)
        return (
            gr.update(interactive=enabled),  # save_history_btn
            gr.update(interactive=enabled),  # load_history_btn
            gr.update(interactive=enabled),  # new_chat_btn
            check_send_button_with_user_id(user_id, system_prompt, logic_history),  # send_button
            gr.update(interactive=enabled),  # reset_button
            gr.update(choices=history_choices),  # history_dropdown
        )

    user_id_input.change(
        update_buttons_on_user_id,
        [user_id_input, system_prompt_input, logic_history_state],
        [
            save_history_btn,
            load_history_btn,
            new_chat_btn,
            send_button,
            reset_button,
            history_dropdown,
        ],
    )

    # Confirmation dialog event handlers
    def handle_confirmation_no():
        """Handle No button click - cancel pending action"""
        return hide_confirmation()

    confirmation_no_btn.click(
        handle_confirmation_no,
        outputs=[confirmation_dialog, confirmation_message, confirmation_state],
    )

    # Yes button handler - execute pending action
    def _execute_save_overwrite(data):
        """Helper to execute save action"""
        status, choices = save_history_action(
            data["user_id"],
            data["save_name"],
            data["logic_hist"],
            data["sys_prompt"],
        )

        # Update token display and send button state after save
        user_id = data["user_id"]
        sys_prompt = data["sys_prompt"]
        logic_hist = data["logic_hist"]
        token_display_value = update_token_display(sys_prompt, logic_hist)
        send_button_state = check_send_button_with_user_id(user_id, sys_prompt, logic_hist)

        return (
            status,
            gr.update(choices=choices),
            gr.update(),  # Don't change chatbot_ui
            gr.update(),  # Keep current display_history
            gr.update(),  # Keep current logic_history
            gr.update(),  # Keep current system_prompt
            token_display_value,  # Update token_display
            send_button_state,  # Update send_button
            *hide_confirmation(),
        )

    def _execute_load_unsaved(data):
        """Helper to execute load action"""
        display_hist, logic_hist, sys_prompt, status = load_history_action(
            data["user_id"], data["history_name"]
        )

        # Check if load failed (returns None)
        if display_hist is None:
            # Load failed, preserve current state but refresh dropdown
            user_id = data["user_id"]
            history_choices = get_history_list(user_id)
            return (
                status,  # Show error message
                gr.update(choices=history_choices),  # Refresh dropdown
                gr.update(),  # Keep current chatbot_ui
                gr.update(),  # Keep current display_history
                gr.update(),  # Keep current logic_history
                gr.update(),  # Keep current system_prompt
                gr.update(),  # Keep current token_display
                gr.update(),  # Keep current send_button
                *hide_confirmation(),
            )

        # Use helper to build consistent UI updates
        user_id = data["user_id"]
        updates = _build_history_operation_updates(
            user_id, display_hist, logic_hist, sys_prompt, status
        )

        return (
            updates["history_status"],
            updates["history_dropdown"],
            updates["chatbot_ui"],
            updates["display_history_state"],
            updates["logic_history_state"],
            updates["system_prompt_input"],
            updates["token_display"],
            updates["send_button"],
            *hide_confirmation(),
        )

    def _execute_new_chat_unsaved(data):
        """Helper to execute new chat action"""
        display_hist, logic_hist, sys_prompt, status = new_chat_action()

        # Use helper to build consistent UI updates
        user_id = data.get("user_id", "")
        updates = _build_history_operation_updates(
            user_id, display_hist, logic_hist, sys_prompt, status
        )

        return (
            updates["history_status"],
            updates["history_dropdown"],
            updates["chatbot_ui"],
            updates["display_history_state"],
            updates["logic_history_state"],
            updates["system_prompt_input"],
            updates["token_display"],
            updates["send_button"],
            *hide_confirmation(),
        )

    ACTION_HANDLERS = {
        "save_overwrite": _execute_save_overwrite,
        "load_unsaved": _execute_load_unsaved,
        "new_chat_unsaved": _execute_new_chat_unsaved,
    }

    def handle_confirmation_yes(conf_state):
        """Handle Yes button click - execute pending action"""
        action = conf_state.get("pending_action")
        handler = ACTION_HANDLERS.get(action)

        if handler:
            data = conf_state.get("pending_data", {})
            return handler(data)

        # Unknown action, just hide dialog
        return (
            "",
            gr.update(),
            gr.update(),  # Don't change chatbot_ui
            gr.update(),  # Don't change display_history
            gr.update(),  # Don't change logic_history
            gr.update(),  # Don't change system_prompt
            gr.update(),  # Don't change token_display
            gr.update(),  # Don't change send_button
            *hide_confirmation(),
        )

    confirmation_yes_btn.click(
        handle_confirmation_yes,
        inputs=[confirmation_state],
        outputs=[
            history_status,
            history_dropdown,
            chatbot_ui,  # Add chatbot_ui to outputs
            display_history_state,
            logic_history_state,
            system_prompt_input,
            token_display,  # Add token_display to outputs
            send_button,  # Add send_button to outputs
            confirmation_dialog,
            confirmation_message,
            confirmation_state,
        ],
    )

    # History operations with confirmation flow
    def handle_save_history(user_id, save_name, display_hist, logic_hist, sys_prompt):
        """Handle save history button click"""
        if not save_name or not save_name.strip():
            return (
                "❌ 保存名を入力してください",
                gr.update(),
                gr.update(),  # Keep current token_display
                gr.update(),  # Keep current send_button
                *hide_confirmation(),
            )

        save_name = save_name.strip()

        # Check if name exists and show confirmation for overwrite
        if check_history_name_exists(user_id, save_name):
            return (
                "",  # Don't update status yet
                gr.update(),  # Don't update dropdown yet
                gr.update(),  # Keep current token_display
                gr.update(),  # Keep current send_button
                *show_confirmation(
                    f"履歴 '{save_name}' は既に存在します。上書きしますか？",
                    "save_overwrite",
                    {
                        "user_id": user_id,
                        "save_name": save_name,
                        "display_hist": display_hist,
                        "logic_hist": logic_hist,
                        "sys_prompt": sys_prompt,
                    },
                ),
            )

        # No conflict, save directly
        status, choices = save_history_action(user_id, save_name, logic_hist, sys_prompt)

        # Update token display and send button state after save
        token_display_value = update_token_display(sys_prompt, logic_hist)
        send_button_state = check_send_button_with_user_id(user_id, sys_prompt, logic_hist)

        return (
            status,
            gr.update(choices=choices),
            token_display_value,
            send_button_state,
            *hide_confirmation(),
        )

    save_history_btn.click(
        handle_save_history,
        inputs=[
            user_id_input,
            save_name_input,
            display_history_state,
            logic_history_state,
            system_prompt_input,
        ],
        outputs=[
            history_status,
            history_dropdown,
            token_display,  # Update token count
            send_button,  # Update send button state
            confirmation_dialog,
            confirmation_message,
            confirmation_state,
        ],
    )

    def handle_load_history(user_id, history_name, logic_hist):
        """Handle load history button click"""
        if not history_name:
            return (
                gr.update(),  # Don't change chatbot_ui
                gr.update(),  # Don't change display_history
                gr.update(),  # Don't change logic_history
                gr.update(),  # Don't change system_prompt
                "❌ 読み込む履歴を選択してください",
                gr.update(),  # Don't change token_display
                gr.update(),  # Don't change send_button
                gr.update(),  # Don't change history_dropdown
                *hide_confirmation(),
            )

        # Check for unsaved session and show confirmation
        if has_unsaved_session(logic_hist):
            return (
                gr.update(),  # Keep current chatbot_ui
                gr.update(),  # Keep current display_history
                gr.update(),  # Keep current logic_history
                gr.update(),  # Keep current system_prompt
                "",  # Don't update status yet
                gr.update(),  # Keep current token_display
                gr.update(),  # Keep current send_button
                gr.update(),  # Keep current history_dropdown
                *show_confirmation(
                    "未保存の会話があります。破棄して読み込みますか？",
                    "load_unsaved",
                    {"user_id": user_id, "history_name": history_name},
                ),
            )

        # No unsaved content, load directly
        display_hist, logic_hist, sys_prompt, status = load_history_action(user_id, history_name)

        # Check if load failed (returns None)
        if display_hist is None:
            # Load failed, preserve current state but refresh dropdown
            history_choices = get_history_list(user_id)
            return (
                gr.update(),  # Keep current chatbot_ui
                gr.update(),  # Keep current display_history
                gr.update(),  # Keep current logic_history
                gr.update(),  # Keep current system_prompt
                status,  # Show error message
                gr.update(),  # Keep current token_display
                gr.update(),  # Keep current send_button
                gr.update(choices=history_choices),  # Refresh dropdown
                *hide_confirmation(),
            )

        # Use helper to build consistent UI updates
        updates = _build_history_operation_updates(
            user_id, display_hist, logic_hist, sys_prompt, status
        )

        return (
            updates["chatbot_ui"],
            updates["display_history_state"],
            updates["logic_history_state"],
            updates["system_prompt_input"],
            updates["history_status"],
            updates["token_display"],
            updates["send_button"],
            updates["history_dropdown"],
            *hide_confirmation(),
        )

    load_history_btn.click(
        handle_load_history,
        inputs=[user_id_input, history_dropdown, logic_history_state],
        outputs=[
            chatbot_ui,  # Update chatbot display
            display_history_state,
            logic_history_state,
            system_prompt_input,
            history_status,
            token_display,  # Update token count
            send_button,  # Update send button state
            history_dropdown,  # Update history list
            confirmation_dialog,
            confirmation_message,
            confirmation_state,
        ],
    )

    def handle_new_chat(user_id, logic_hist, sys_prompt):
        """Handle new chat button click"""
        # Check for unsaved session and show confirmation
        if has_unsaved_session(logic_hist):
            return (
                gr.update(),  # Keep current chatbot_ui
                gr.update(),  # Keep current display_history
                gr.update(),  # Keep current logic_history
                gr.update(),  # Keep current system_prompt
                "",  # Don't update status yet
                gr.update(),  # Keep current token_display
                gr.update(),  # Keep current send_button
                *show_confirmation(
                    "未保存の会話があります。破棄して新規開始しますか？",
                    "new_chat_unsaved",
                    {"user_id": user_id},  # Store user_id for token/button update
                ),
            )

        # No unsaved content, start new chat directly
        display_hist, logic_hist, sys_prompt, status = new_chat_action()

        # Use helper to build consistent UI updates
        updates = _build_history_operation_updates(
            user_id, display_hist, logic_hist, sys_prompt, status
        )

        return (
            updates["chatbot_ui"],
            updates["display_history_state"],
            updates["logic_history_state"],
            updates["system_prompt_input"],
            updates["history_status"],
            updates["token_display"],
            updates["send_button"],
            *hide_confirmation(),
        )

    # Common inputs and outputs for new chat action
    new_chat_inputs = [user_id_input, logic_history_state, system_prompt_input]
    new_chat_outputs = [
        chatbot_ui,
        display_history_state,
        logic_history_state,
        system_prompt_input,
        history_status,
        token_display,
        send_button,
        confirmation_dialog,
        confirmation_message,
        confirmation_state,
    ]

    # Both new_chat_btn and reset_button trigger the same handler
    new_chat_btn.click(handle_new_chat, inputs=new_chat_inputs, outputs=new_chat_outputs)
    reset_button.click(handle_new_chat, inputs=new_chat_inputs, outputs=new_chat_outputs)

    # Update token display and button state when system prompt or history changes
    system_prompt_input.change(
        lambda user_id, prompt, history: (
            update_token_display(prompt, history),
            check_send_button_with_user_id(user_id, prompt, history),
        ),
        [user_id_input, system_prompt_input, logic_history_state],
        [token_display, send_button],
    )

    # イベントハンドラを定義（user_inputとsend_buttonの両方）
    submit_inputs = [
        user_input,
        display_history_state,
        logic_history_state,
        system_prompt_input,
        user_id_input,
    ]
    submit_outputs = [chatbot_ui, display_history_state, logic_history_state]

    # Update token display and send button state after response
    def update_token_and_button(user_id, logic, sys):
        """Update token display and button state after response (success or error)"""
        return (
            update_token_display(sys, logic),
            check_send_button_with_user_id(user_id, sys, logic),
        )

    user_input.submit(respond, submit_inputs, submit_outputs).then(
        update_token_and_button,
        [user_id_input, logic_history_state, system_prompt_input],
        [token_display, send_button],
    )
    send_button.click(respond, submit_inputs, submit_outputs).then(
        update_token_and_button,
        [user_id_input, logic_history_state, system_prompt_input],
        [token_display, send_button],
    )

    # 送信後、入力ボックスをクリアする
    user_input.submit(lambda: "", None, user_input)
    send_button.click(lambda: "", None, user_input)


def launch(server_name=None, debug=True):
    """Launch the Gradio demo with env-aware defaults."""
    resolved_server = (
        os.getenv("MLC_SERVER_NAME", "127.0.0.1") if server_name is None else server_name
    )
    demo.launch(server_name=resolved_server, debug=debug)
