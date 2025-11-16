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


def update_token_display(system_prompt, model_name=None):
    """Update token count display for system prompt"""
    if model_name is None:
        # Use smallest context length to be conservative
        model_name = core.CHATGPT_MODEL

    if not system_prompt:
        return "Tokens: 0 / - (no system prompt)"

    token_info = core.get_token_info(system_prompt, model_name)
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


def check_send_button_enabled(system_prompt, model_name=None):
    """Check if send button should be enabled based on token limit"""
    if model_name is None:
        # Use smallest context length to be conservative
        model_name = core.CHATGPT_MODEL

    if not system_prompt:
        return gr.Button(interactive=True)

    token_info = core.get_token_info(system_prompt, model_name)
    is_enabled = token_info["token_count"] <= token_info["max_context_length"]
    return gr.Button(interactive=is_enabled)


def respond(user_message, display_history, logic_history, system_prompt):
    """
    ユーザー入力への応答、LLM呼び出し、履歴管理をすべて行う単一の関数。
    """

    def _stream_response(model_name, stream):
        full_response = ""
        for chunk in stream:
            text = ""
            try:
                if model_name == "gemini":
                    text = chunk.text
                elif model_name == "chatgpt":
                    delta_content = chunk.choices[0].delta.content
                    # Handle both string and list responses from OpenAI API
                    if isinstance(delta_content, list):
                        text = "".join(
                            part.text if hasattr(part, "text") else str(part)
                            for part in delta_content
                        )
                    elif delta_content is not None:
                        text = delta_content
            except (AttributeError, IndexError, TypeError, ValueError):
                if isinstance(chunk, str):
                    text = chunk

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


# --- Gradio UIの構築 ---
with gr.Blocks() as demo:
    gr.Markdown("# Multi-LLM Chat")

    # System prompt input
    with gr.Row():
        system_prompt_input = gr.Textbox(
            label="System Prompt",
            placeholder="Enter system prompt (optional)...",
            lines=3,
        )

    # Token count display
    token_display = gr.Markdown("Tokens: 0 / - (no system prompt)")

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
        send_button = gr.Button("Send", variant="primary", scale=1)

    # Update token display and button state when system prompt changes
    system_prompt_input.change(
        lambda prompt: (update_token_display(prompt), check_send_button_enabled(prompt)),
        [system_prompt_input],
        [token_display, send_button],
    )

    # イベントハンドラを定義（user_inputとsend_buttonの両方）
    submit_inputs = [user_input, display_history_state, logic_history_state, system_prompt_input]
    submit_outputs = [chatbot_ui, display_history_state, logic_history_state]

    user_input.submit(respond, submit_inputs, submit_outputs)
    send_button.click(respond, submit_inputs, submit_outputs)

    # 送信後、入力ボックスをクリアする
    user_input.submit(lambda: "", None, user_input)
    send_button.click(lambda: "", None, user_input)


def launch(server_name=None, debug=True):
    """Launch the Gradio demo with env-aware defaults."""
    resolved_server = (
        os.getenv("MLC_SERVER_NAME", "127.0.0.1") if server_name is None else server_name
    )
    demo.launch(server_name=resolved_server, debug=debug)
