import gradio as gr
import time

from gradio_client import utils as gradio_utils


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

# Gradioのバグを回避するため、ロジックのインポートは
# UIイベントがトリガーされる関数内で行う（遅延インポート）

def respond(user_message, display_history, logic_history):
    """
    ユーザー入力への応答、LLM呼び出し、履歴管理をすべて行う単一の関数。
    この関数はジェネレータであり、UIをストリーミング更新するために
    表示用の会話履歴(display_history)をyieldで繰り返し返す。
    """
    # --- 遅延インポート ---
    from chat_logic import call_gemini_api, call_chatgpt_api

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

    # 3. Geminiへの応答処理
    if mention in ["gemini", "all"]:
        display_history[-1][1] = ""
        full_response = ""
        stream = call_gemini_api(logic_history)
        for chunk in stream:
            try:
                if chunk.text:
                    text = chunk.text
                    full_response += text
                    display_history[-1][1] = "**Gemini:**\n" + full_response
                    yield display_history, display_history, logic_history
            except (AttributeError, TypeError):
                if isinstance(chunk, str):
                    full_response += chunk
                    display_history[-1][1] = "**Gemini:**\n" + full_response
                    yield display_history, display_history, logic_history
        
        logic_history.append({"role": "gemini", "content": full_response})
        if not full_response.strip():
             display_history[-1][1] = "**Gemini:**\n[System: Geminiからの応答がありませんでした]"
        yield display_history, display_history, logic_history

    # 4. ChatGPTへの応答処理
    if mention in ["chatgpt", "all"]:
        # @all の場合、表示を新しい行に切り替える
        if mention == "all":
            display_history.append([user_message, ""])
        else:
            display_history[-1][1] = ""
        
        full_response = ""
        stream = call_chatgpt_api(logic_history)
        for chunk in stream:
            try:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    display_history[-1][1] = "**ChatGPT:**\n" + full_response
                    yield display_history, display_history, logic_history
            except (AttributeError, IndexError, TypeError):
                 if isinstance(chunk, str):
                    full_response += chunk
                    display_history[-1][1] = "**ChatGPT:**\n" + full_response
                    yield display_history, display_history, logic_history

        logic_history.append({"role": "chatgpt", "content": full_response})
        if not full_response.strip():
             display_history[-1][1] = "**ChatGPT:**\n[System: ChatGPTからの応答がありませんでした]"
        yield display_history, display_history, logic_history

# --- Gradio UIの構築 ---
with gr.Blocks() as demo:
    gr.Markdown("# Multi-LLM Chat")
    
    # 履歴を管理するための非表示Stateコンポーネント
    display_history_state = gr.State([])
    logic_history_state = gr.State([])

    # UIコンポーネント
    chatbot_ui = gr.Chatbot(label="Conversation", height=600)
    user_input = gr.Textbox(show_label=False, placeholder="Enter text with @mention...", container=False)

    # イベントハンドラを定義
    user_input.submit(
        respond, # メインの応答関数
        [user_input, display_history_state, logic_history_state], # 入力
        [chatbot_ui, display_history_state, logic_history_state]  # 出力
    )
    # 送信後、入力ボックスをクリアする
    user_input.submit(lambda: "", None, user_input)

if __name__ == "__main__":
    demo.launch(debug=True)
