"""UI state management for WebUI"""

import gradio as gr

from .. import core


class WebUIState:
    """Manages the state of the Gradio UI components."""

    def __init__(
        self,
        user_id: str | None,
        has_history: bool,
        is_streaming: bool,
        system_prompt: str | None = None,
        logic_history: list | None = None,
        model_name: str | None = None,
    ):
        """
        Args:
            user_id: The current user ID.
            has_history: Whether history exists for the current user.
            is_streaming: Whether a response is currently being streamed.
            system_prompt: The system prompt text.
            logic_history: The current conversation history.
            model_name: The name of the model for context length calculation.
        """
        self.user_id = user_id
        self.has_history = has_history
        self.is_streaming = is_streaming
        self.system_prompt = system_prompt
        self.logic_history = logic_history
        self.model_name = model_name or core.CHATGPT_MODEL

    def get_button_states(self) -> dict:
        """
        Determines the interactive state of all buttons based on the current UI state.

        Returns:
            A dictionary mapping button component names to gr.update() objects.
        """
        user_id_exists = bool(self.user_id and self.user_id.strip())
        can_interact = user_id_exists and not self.is_streaming

        # Check token limit for send button
        send_enabled = can_interact
        if self.system_prompt and can_interact:
            token_info = core.get_token_info(
                self.system_prompt, self.model_name, self.logic_history
            )
            send_enabled = token_info["token_count"] <= token_info["max_context_length"]

        can_save = can_interact and bool(self.logic_history)

        # Return button states in consistent order
        return {
            "send_button": gr.update(interactive=send_enabled),
            "new_chat_btn": gr.update(interactive=can_interact),
            "reset_button": gr.update(interactive=can_interact),
            "save_history_btn": gr.update(interactive=can_save),
            "load_history_btn": gr.update(interactive=can_interact and self.has_history),
            "system_prompt_save_btn": gr.update(interactive=can_interact),
        }


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
