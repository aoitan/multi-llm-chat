from . import core


def _clone_history(history):
    """Create a shallow copy of the conversation history."""
    return [entry.copy() for entry in history]


def _process_response_stream(stream, model_name):
    """Helper function to process and print response streams from LLMs."""
    full_response = ""
    print(f"[{model_name.capitalize()}]: ", end="", flush=True)

    for chunk in stream:
        text = ""
        try:
            if model_name == "gemini":
                text = chunk.text
            elif model_name == "chatgpt":
                text = chunk.choices[0].delta.content
        except (AttributeError, IndexError, TypeError, ValueError):
            if isinstance(chunk, str):
                text = chunk

        if text:
            print(text, end="", flush=True)
            full_response += text

    print()
    if not full_response.strip():
        print(
            f"[System: {model_name.capitalize()}からの応答がありませんでした。"
            f"プロンプトがブロックされた可能性があります。]",
            flush=True,
        )

    return full_response


def _handle_system_command(args, system_prompt, current_model=None):
    """Handle /system command"""
    if current_model is None:
        current_model = core.GEMINI_MODEL

    if not args:
        # Display current system prompt
        if system_prompt:
            print(f"現在のシステムプロンプト: {system_prompt}")
        else:
            print("システムプロンプトは設定されていません。")
        return system_prompt

    if args == "clear":
        # Clear system prompt
        print("システムプロンプトをクリアしました。")
        return ""

    # Set new system prompt
    new_prompt = args
    token_info = core.get_token_info(new_prompt, current_model)

    if token_info["token_count"] > token_info["max_context_length"]:
        print(
            f"警告: トークン数が上限を超えています。"
            f"({token_info['token_count']} / {token_info['max_context_length']}) "
            f"設定できません。"
        )
        return system_prompt

    print(f"システムプロンプトを設定しました: {new_prompt}")
    return new_prompt


def main():
    """Main CLI loop"""
    history = []
    system_prompt = ""

    while True:
        prompt = input("> ").strip()

        if prompt.lower() in ["exit", "quit"]:
            break

        # Handle commands
        if prompt.startswith("/"):
            parts = prompt.split(None, 1)
            command = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            if command == "/system":
                system_prompt = _handle_system_command(args, system_prompt)
            else:
                print(
                    f"エラー: `{command}` は不明なコマンドです。"
                    f"利用可能なコマンドは `/system` などです。"
                )
            continue

        history.append({"role": "user", "content": prompt})

        if prompt.startswith("@gemini"):
            gemini_stream = core.call_gemini_api(history, system_prompt)
            response_g = _process_response_stream(gemini_stream, "gemini")
            history.append({"role": "gemini", "content": response_g})

        elif prompt.startswith("@chatgpt"):
            chatgpt_stream = core.call_chatgpt_api(history, system_prompt)
            response_c = _process_response_stream(chatgpt_stream, "chatgpt")
            history.append({"role": "chatgpt", "content": response_c})

        elif prompt.startswith("@all"):
            shared_history = _clone_history(history)

            gemini_stream = core.call_gemini_api(shared_history, system_prompt)
            response_g = _process_response_stream(gemini_stream, "gemini")
            history.append({"role": "gemini", "content": response_g})

            chatgpt_stream = core.call_chatgpt_api(shared_history, system_prompt)
            response_c = _process_response_stream(chatgpt_stream, "chatgpt")
            history.append({"role": "chatgpt", "content": response_c})

        else:
            # Thinking memo
            pass

    return history, system_prompt
