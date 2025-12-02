import os

import gradio as gr
from gradio_client import utils as gradio_utils

from . import core

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


def check_send_button_enabled(system_prompt, logic_history=None, model_name=None):
    """Check if send button should be enabled based on token limit

    Args:
        system_prompt: System prompt text
        logic_history: Current conversation history (optional)
        model_name: Model name for context length calculation (optional)

    Returns:
        gr.Button with interactive state set
    """
    if model_name is None:
        # Use smallest context length to be conservative
        model_name = core.CHATGPT_MODEL

    if not system_prompt:
        return gr.Button(interactive=True)

    # Include history in token calculation
    token_info = core.get_token_info(system_prompt, model_name, logic_history)
    is_enabled = token_info["token_count"] <= token_info["max_context_length"]
    return gr.Button(interactive=is_enabled)


def respond(user_message, display_history, logic_history, system_prompt, user_id=None):
    """
    ユーザー入力への応答、LLM呼び出し、履歴管理をすべて行う単一の関数。

    Args:
        user_message: User's input message
        display_history: Display history for chatbot UI
        logic_history: Internal logic history
        system_prompt: System prompt text
        user_id: User ID (required for history management)
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

    # 履歴を管理するための非表示Stateコンポーネント
    display_history_state = gr.State([])
    logic_history_state = gr.State([])

    # UIコンポーネント
    chatbot_ui = gr.Chatbot(label="Conversation", height=600)

    with gr.Row():
        user_input = gr.Textbox(
            show_label=False,
            placeholder="Enter text with @mention...",
            container=False,
            scale=4,
        )
        send_button = gr.Button("Send", variant="primary", scale=1, interactive=False)

    # Update button states when user ID changes
    def update_buttons_on_user_id(user_id, system_prompt, logic_history):
        enabled = bool(user_id and user_id.strip())
        return (
            gr.update(interactive=enabled),  # save_history_btn
            gr.update(interactive=enabled),  # load_history_btn
            gr.update(interactive=enabled),  # new_chat_btn
            check_send_button_with_user_id(user_id, system_prompt, logic_history),  # send_button
        )

    user_id_input.change(
        update_buttons_on_user_id,
        [user_id_input, system_prompt_input, logic_history_state],
        [save_history_btn, load_history_btn, new_chat_btn, send_button],
    )

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

    # Remove unused function (dead code)
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
