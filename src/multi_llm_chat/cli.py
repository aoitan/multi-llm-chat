import os
import sys

from . import core
from .history import HistoryStore, sanitize_name


def _clone_history(history):
    """Create a shallow copy of the conversation history."""
    return [entry.copy() for entry in history]


def _process_response_stream(stream, model_name):
    """Helper function to process and print response streams from LLMs."""
    full_response = ""
    print(f"[{model_name.capitalize()}]: ", end="", flush=True)

    for chunk in stream:
        text = core.extract_text_from_chunk(chunk, model_name)
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
    """Handle /system command

    Uses the largest available context window across enabled models to avoid
    unnecessarily rejecting prompts that would work with some models.
    """
    if current_model is None:
        # Use the largest context length across enabled models
        # This allows users to set prompts for high-context models (Gemini)
        # while still warning if they exceed all model limits
        current_model = core.GEMINI_MODEL  # Gemini has the largest context

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


def _prompt_user_id():
    """Prompt for a user ID with spoofing warning.

    If CHAT_HISTORY_USER_ID is set, use it to keep non-interactive flows working.
    """
    env_user_id = os.getenv("CHAT_HISTORY_USER_ID")
    if env_user_id:
        try:
            sanitize_name(env_user_id)
            return env_user_id
        except ValueError:
            print("エラー: CHAT_HISTORY_USER_ID が無効です。入力し直してください。")
            env_user_id = None

    prompt_text = "ユーザーIDを入力してください: "

    # Non-interactive fallback: do not consume stdin (keeps legacy piping intact)
    if not sys.stdin.isatty():
        return env_user_id or "default"

    print("警告: これはユーザー認証ではありません。ローカルテスト用途のIDを入力してください。")

    while True:
        user_id = input(prompt_text).strip()
        try:
            sanitize_name(user_id)
            return user_id
        except ValueError:
            print(
                "エラー: 無効な名前です。許可される文字は"
                "英数字、ハイフン、アンダースコアのみです。"
            )


def _confirm(prompt):
    """Ask the user for a yes/no confirmation."""
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def _handle_history_list(user_id, store):
    """Handle the '/history list' command."""
    names = store.list_histories(user_id)
    if not names:
        print("保存済みの履歴はありません。")
    else:
        print("保存済みの履歴:")
        for name in names:
            print(f"- {name}")
    return None, None, None


def _handle_history_new(is_dirty):
    """Handle the '/history new' command."""
    if is_dirty and not _confirm("現在の会話は保存されていません。破棄しますか？"):
        return None, None, None
    return [], "", False


def _handle_history_save(args, user_id, store, system_prompt, history):
    """Handle the '/history save <name>' command."""
    name = args
    if not name:
        print("エラー: 保存名を指定してください。")
        return None, None, None

    try:
        sanitize_name(name)
    except ValueError:
        print("エラー: 無効な名前です。許可される文字は英数字、ハイフン、アンダースコアのみです。")
        return None, None, None

    if store.history_exists(user_id, name):
        if not _confirm("同じ名前の履歴が存在します。上書きしますか？"):
            return None, None, None

    try:
        store.save_history(user_id, name, system_prompt, history)
        print(f"履歴を保存しました: {name}")
        return history, system_prompt, False
    except OSError:
        print("エラー: 履歴の保存に失敗しました。")
        return None, None, None


def _handle_history_load(args, user_id, store, is_dirty):
    """Handle the '/history load <name>' command."""
    name = args
    if not name:
        print("エラー: 読み込む履歴名を指定してください。")
        return None, None, None

    try:
        sanitize_name(name)
    except ValueError:
        print("エラー: 無効な名前です。許可される文字は英数字、ハイフン、アンダースコアのみです。")
        return None, None, None

    if is_dirty and not _confirm("現在の会話は保存されていません。新しい履歴を読み込みますか？"):
        return None, None, None

    try:
        loaded = store.load_history(user_id, name)
        print(f"履歴を読み込みました: {name}")
        return loaded.get("turns", []), loaded.get("system_prompt", ""), False
    except FileNotFoundError:
        print("エラー: 指定された履歴が見つかりませんでした。現在の会話は変更されていません。")
    except OSError:
        print("エラー: 履歴の読み込みに失敗しました。現在の会話は変更されていません。")
    return None, None, None


def _handle_history_command(
    command_args,
    user_id,
    store,
    history,
    system_prompt,
    is_dirty,
):
    """Handle history-related CLI commands."""
    if not command_args:
        print(
            "履歴コマンドの使い方: /history list | /history save <名前> | "
            "/history load <名前> | /history new"
        )
        return history, system_prompt, is_dirty

    parts = command_args.split(None, 1)
    action = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    new_history, new_system_prompt, new_is_dirty = None, None, None

    if action == "list":
        new_history, new_system_prompt, new_is_dirty = _handle_history_list(
            user_id, store
        )
    elif action == "new":
        new_history, new_system_prompt, new_is_dirty = _handle_history_new(is_dirty)
    elif action == "save":
        new_history, new_system_prompt, new_is_dirty = _handle_history_save(
            args, user_id, store, system_prompt, history
        )
    elif action == "load":
        new_history, new_system_prompt, new_is_dirty = _handle_history_load(
            args, user_id, store, is_dirty
        )
    else:
        print(
            f"エラー: `{action}` は不明な履歴コマンドです。"
            f"利用可能なコマンドは list/save/load/new です。"
        )

    return (
        new_history if new_history is not None else history,
        new_system_prompt if new_system_prompt is not None else system_prompt,
        new_is_dirty if new_is_dirty is not None else is_dirty,
    )


def main():
    """Main CLI loop"""
    store = HistoryStore()
    user_id = _prompt_user_id()

    history = []
    system_prompt = ""
    is_dirty = False

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
                new_prompt = _handle_system_command(args, system_prompt)
                if new_prompt != system_prompt:
                    system_prompt = new_prompt
                    is_dirty = True
            elif command == "/history":
                history, system_prompt, is_dirty = _handle_history_command(
                    args, user_id, store, history, system_prompt, is_dirty
                )
            else:
                print(
                    f"エラー: `{command}` は不明なコマンドです。"
                    f"利用可能なコマンドは `/system` などです。"
                )
            continue

        history.append({"role": "user", "content": prompt})
        is_dirty = True

        if prompt.startswith("@gemini"):
            gemini_stream = core.call_gemini_api(history, system_prompt)
            response_g = _process_response_stream(gemini_stream, "gemini")
            history.append({"role": "gemini", "content": response_g})
            is_dirty = True

        elif prompt.startswith("@chatgpt"):
            chatgpt_stream = core.call_chatgpt_api(history, system_prompt)
            response_c = _process_response_stream(chatgpt_stream, "chatgpt")
            history.append({"role": "chatgpt", "content": response_c})
            is_dirty = True

        elif prompt.startswith("@all"):
            shared_history = _clone_history(history)

            gemini_stream = core.call_gemini_api(shared_history, system_prompt)
            response_g = _process_response_stream(gemini_stream, "gemini")
            history.append({"role": "gemini", "content": response_g})
            is_dirty = True

            chatgpt_stream = core.call_chatgpt_api(shared_history, system_prompt)
            response_c = _process_response_stream(chatgpt_stream, "chatgpt")
            history.append({"role": "chatgpt", "content": response_c})
            is_dirty = True

        else:
            # Thinking memo
            pass

    return history, system_prompt
