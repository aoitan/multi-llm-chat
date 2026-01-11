"""Gradio app builder for WebUI"""

import logging
import os

import gradio as gr

from .components import update_token_display
from .handlers import (
    check_history_name_exists,
    get_history_list,
    has_history_for_user,
    has_unsaved_session,
    load_history_action,
    logic_history_to_display,
    new_chat_action,
    save_history_action,
    validate_and_respond,
)
from .state import WebUIState, hide_confirmation, show_confirmation

logger = logging.getLogger(__name__)


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
                "選択した会話を読み込む",
                elem_id="load_history_btn",
                interactive=False,
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

    # Session-scoped ChatService for provider reuse (Issue #58)
    chat_service_state = gr.State(None)

    # UIコンポーネント
    chatbot_ui = gr.Chatbot(label="Conversation", height=600, show_copy_button=True)

    with gr.Row():
        user_input = gr.Textbox(
            show_label=False,
            placeholder="Enter text with @mention...",
            container=False,
            scale=4,
        )
        send_button = gr.Button(
            "Send", variant="primary", scale=1, interactive=False, elem_id="send_button"
        )
        reset_button = gr.Button(
            "新しい会話を開始",
            variant="secondary",
            scale=1,
            interactive=False,
            elem_id="reset_button",
        )

    # --- UI Component Mapping ---
    # For updating components dynamically from WebUIState
    # This avoids hardcoding component names in state logic
    button_components = {
        "send_button": send_button,
        "new_chat_btn": new_chat_btn,
        "save_history_btn": save_history_btn,
        "load_history_btn": load_history_btn,
        "reset_button": reset_button,
        "system_prompt_save_btn": gr.Button(
            "Save System Prompt", visible=False
        ),  # Placeholder for future use
    }

    # Update button states when user ID changes
    def update_ui_on_user_id_change(
        user_id: str, logic_history: list, system_prompt: str
    ):
        """
        Updates UI components based on user ID changes.
        Fetches history and uses WebUIState to determine button states and other UI elements.
        """
        history_exists = has_history_for_user(user_id)
        history_choices = get_history_list(user_id)

        # UI state is determined by user_id, history, and token counts
        state = WebUIState(
            user_id=user_id,
            has_history=history_exists,
            is_streaming=False,
            system_prompt=system_prompt,
            logic_history=logic_history,
        )
        button_states = state.get_button_states()

        # Convert logic history to display format for the chatbot UI
        display_history = logic_history_to_display(logic_history)

        # Create a list of UI updates in the correct order for the outputs
        updates = [
            display_history,  # chatbot_ui
            logic_history,  # logic_history_state (no change)
            system_prompt,  # system_prompt_input (no change)
            button_states.get("save_history_btn", gr.update()),
            button_states.get("load_history_btn", gr.update()),
            button_states.get("new_chat_btn", gr.update()),
            button_states.get("send_button", gr.update()),
            button_states.get("reset_button", gr.update()),
            gr.update(choices=history_choices),
            None,  # Reset chat_service_state when user changes
        ]
        return updates

    user_id_input.change(
        update_ui_on_user_id_change,
        inputs=[user_id_input, logic_history_state, system_prompt_input],
        outputs=[
            chatbot_ui,
            logic_history_state,
            system_prompt_input,
            save_history_btn,
            load_history_btn,
            new_chat_btn,
            send_button,
            reset_button,
            history_dropdown,
            chat_service_state,
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
        has_history = has_history_for_user(user_id)
        state = WebUIState(
            user_id=user_id,
            has_history=has_history,
            is_streaming=False,
            system_prompt=sys_prompt,
            logic_history=logic_hist,
        )
        button_states = state.get_button_states()
        token_display_value = update_token_display(sys_prompt, logic_hist)
        send_button_state = button_states["send_button"]

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

        # Use WebUIState to build consistent UI updates
        user_id = data["user_id"]
        has_history = has_history_for_user(user_id)
        state = WebUIState(
            user_id=user_id,
            has_history=has_history,
            is_streaming=False,
            system_prompt=sys_prompt,
            logic_history=logic_hist,
        )
        button_states = state.get_button_states()
        token_display_value = update_token_display(sys_prompt, logic_hist)
        history_choices = get_history_list(user_id)

        return (
            status,
            gr.update(choices=history_choices),
            display_hist,  # chatbot_ui
            display_hist,  # display_history_state
            logic_hist,  # logic_history_state
            sys_prompt,  # system_prompt_input
            token_display_value,
            button_states["send_button"],
            *hide_confirmation(),
        )

    def _execute_new_chat_unsaved(data):
        """Helper to execute new chat action"""
        display_hist, logic_hist, sys_prompt, status = new_chat_action()

        # Use WebUIState to build consistent UI updates
        user_id = data.get("user_id", "")
        has_history = has_history_for_user(user_id) if user_id else False
        state = WebUIState(
            user_id=user_id,
            has_history=has_history,
            is_streaming=False,
            system_prompt=sys_prompt,
            logic_history=logic_hist,
        )
        button_states = state.get_button_states()
        token_display_value = update_token_display(sys_prompt, logic_hist)

        return (
            status,
            gr.update(),  # Don't update dropdown on new chat
            display_hist,
            display_hist,
            logic_hist,
            sys_prompt,
            token_display_value,
            button_states["send_button"],
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
        has_history = has_history_for_user(user_id)
        state = WebUIState(
            user_id=user_id,
            has_history=has_history,
            is_streaming=False,
            system_prompt=sys_prompt,
            logic_history=logic_hist,
        )
        button_states = state.get_button_states()
        token_display_value = update_token_display(sys_prompt, logic_hist)

        return (
            status,
            gr.update(choices=choices),
            token_display_value,
            button_states["send_button"],
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

        # Use WebUIState to build consistent UI updates
        has_history = has_history_for_user(user_id)
        state = WebUIState(
            user_id=user_id,
            has_history=has_history,
            is_streaming=False,
            system_prompt=sys_prompt,
            logic_history=logic_hist,
        )
        button_states = state.get_button_states()
        token_display_value = update_token_display(sys_prompt, logic_hist)
        history_choices = get_history_list(user_id)

        return (
            display_hist,  # chatbot_ui
            display_hist,  # display_history_state
            logic_hist,  # logic_history_state
            sys_prompt,  # system_prompt_input
            status,
            token_display_value,
            button_states["send_button"],
            gr.update(choices=history_choices),
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

        # Use WebUIState to build consistent UI updates
        has_history = has_history_for_user(user_id) if user_id else False
        state = WebUIState(
            user_id=user_id,
            has_history=has_history,
            is_streaming=False,
            system_prompt=sys_prompt,
            logic_history=logic_hist,
        )
        button_states = state.get_button_states()
        token_display_value = update_token_display(sys_prompt, logic_hist)

        return (
            display_hist,
            display_hist,
            logic_hist,
            sys_prompt,
            status,
            token_display_value,
            button_states["send_button"],
            gr.update(),  # Don't update history_dropdown
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
        history_dropdown,
        confirmation_dialog,
        confirmation_message,
        confirmation_state,
    ]

    # Both new_chat_btn and reset_button trigger the same handler
    new_chat_btn.click(handle_new_chat, inputs=new_chat_inputs, outputs=new_chat_outputs)
    reset_button.click(handle_new_chat, inputs=new_chat_inputs, outputs=new_chat_outputs)

    # Update token display and button state when system prompt or history changes
    def update_ui_on_prompt_change(user_id, prompt, history):
        """Updates token display and send button when system prompt changes."""
        has_history = has_history_for_user(user_id)
        state = WebUIState(
            user_id=user_id,
            has_history=has_history,
            is_streaming=False,
            system_prompt=prompt,
            logic_history=history,
        )
        button_states = state.get_button_states()
        token_display_value = update_token_display(prompt, history)
        return token_display_value, button_states["send_button"]

    system_prompt_input.change(
        update_ui_on_prompt_change,
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
        chat_service_state,
    ]
    
    ordered_buttons = [
        send_button,
        new_chat_btn,
        save_history_btn,
        load_history_btn,
        reset_button,
        button_components["system_prompt_save_btn"],
    ]
    submit_outputs = [
        user_input,
        chatbot_ui,
        display_history_state,
        logic_history_state,
        chat_service_state,
        token_display,
        *ordered_buttons,
    ]

    def handle_chat_submission(
        user_message, display_history, logic_history, system_prompt, user_id, chat_service
    ):
        """
        Handles chat submission, including UI state updates.
        This generator centralizes all UI state changes during a chat response.
        """
        # 1. Disable UI components at the start of streaming
        streaming_start_state = WebUIState(
            user_id=user_id,
            is_streaming=True,
            has_history=has_history_for_user(user_id),
            system_prompt=system_prompt,
            logic_history=logic_history,
        )
        button_updates = streaming_start_state.get_button_states()
        
        yield (
            gr.update(value="", interactive=False),  # user_input
            gr.update(),  # chatbot_ui
            gr.update(),  # display_history_state
            gr.update(),  # logic_history_state
            gr.update(),  # chat_service_state
            gr.update(),  # token_display
            button_updates["send_button"],
            button_updates["new_chat_btn"],
            button_updates["save_history_btn"],
            button_updates["load_history_btn"],
            button_updates["reset_button"],
            button_updates["system_prompt_save_btn"],
        )

        # 2. Stream the response from the core logic
        final_logic_history = logic_history
        for res_display, res_disp_state, res_logic, res_service in validate_and_respond(
            user_message, display_history, logic_history, system_prompt, user_id, chat_service
        ):
            final_logic_history = res_logic
            yield (
                gr.update(),  # user_input
                res_display,  # chatbot_ui
                res_disp_state,  # display_history_state
                res_logic,  # logic_history_state
                res_service,  # chat_service_state
                gr.update(),  # token_display
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),  # buttons
            )

        # 3. Re-enable UI components and update token count at the end
        streaming_end_state = WebUIState(
            user_id=user_id,
            is_streaming=False,
            has_history=has_history_for_user(user_id),
            system_prompt=system_prompt,
            logic_history=final_logic_history,
        )
        button_updates = streaming_end_state.get_button_states()
        token_display_value = update_token_display(system_prompt, final_logic_history)
        
        yield (
            gr.update(interactive=True),  # user_input
            gr.update(),  # chatbot_ui
            gr.update(),  # display_history_state
            final_logic_history,  # logic_history_state (final update)
            gr.update(),  # chat_service_state
            token_display_value,  # token_display
            button_updates["send_button"],
            button_updates["new_chat_btn"],
            button_updates["save_history_btn"],
            button_updates["load_history_btn"],
            button_updates["reset_button"],
            button_updates["system_prompt_save_btn"],
        )

    # Chain events
    (
        user_input.submit(
            lambda: "",
            inputs=None,
            outputs=[user_input],
        ).then(handle_chat_submission, submit_inputs, submit_outputs)
    )

    (
        send_button.click(
            lambda: "",
            inputs=None,
            outputs=[user_input],
        ).then(handle_chat_submission, submit_inputs, submit_outputs)
    )


def launch(server_name=None, debug=True):
    """Launch the Gradio demo with env-aware defaults."""
    resolved_server = (
        os.getenv("MLC_SERVER_NAME", "127.0.0.1") if server_name is None else server_name
    )
    demo.launch(server_name=resolved_server, debug=debug)
