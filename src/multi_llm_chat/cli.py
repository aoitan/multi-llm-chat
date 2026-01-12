import os
import sys

import pyperclip

from . import core
from .chat_logic import ChatService
from .history import HistoryStore, get_llm_response, reset_history, sanitize_name


def _display_tool_response(response_type, content):
    """Display tool call or tool result with visual markers.

    Args:
        response_type: "tool_call" or "tool_result"
        content: Tool call or result content dict
    """
    if response_type == "tool_call":
        name = content.get("name", "unknown")
        args = content.get("arguments", {})
        print(f"\n[Tool Call: {name}]", flush=True)
        if args:
            print(f"  Args: {args}", flush=True)
    elif response_type == "tool_result":
        name = content.get("name", "unknown")
        result_content = content.get("content", "")
        print(f"[Tool Result: {name}]", flush=True)
        print(f"  {result_content}", flush=True)
        print()  # Blank line


async def _process_service_stream(service, user_message):
    """Process ChatService stream and print responses for CLI with real-time streaming.

    Args:
        service: ChatService instance
        user_message: User's input message

    Returns:
        tuple: (display_history, logic_history) after processing
    """
    display_hist, logic_hist = [], []
    last_printed_idx = 0  # Track how many exchanges have been fully printed
    last_content_length = 0  # Track printed content length for current streaming message
    last_model_name = None  # Track model name to detect switches (for @all)

    # Process message through ChatService with streaming
    async for display_hist, logic_hist in service.process_message(user_message):  # noqa: B007
        # Print only new content from display_history (incremental streaming)
        # display_history format: [[user_msg, assistant_msg], ...]

        if not display_hist:
            continue

        # Check if there's a new exchange (new user message added)
        if len(display_hist) > last_printed_idx:
            # New exchange started - reset content length tracker
            last_content_length = 0
            last_printed_idx = len(display_hist)

        # Get the latest assistant message (currently streaming)
        _user_msg, assistant_msg = display_hist[-1]

        if not assistant_msg:
            continue

        # Extract model name and content from Markdown-formatted response
        # Format: "**Gemini:**\nActual content" or "**ChatGPT:**\nContent"
        if assistant_msg.startswith("**Gemini:**"):
            model_name = "Gemini"
            full_content = assistant_msg[len("**Gemini:**\n") :]
        elif assistant_msg.startswith("**ChatGPT:**"):
            model_name = "ChatGPT"
            full_content = assistant_msg[len("**ChatGPT:**\n") :]
        else:
            # Fallback for unexpected format
            model_name = "Assistant"
            full_content = assistant_msg

        # Detect model switch (e.g., @all: Gemini -> ChatGPT)
        if last_model_name is not None and model_name != last_model_name:
            # Add newline to separate different models' responses
            print()
            last_content_length = 0

        # Print only the new part (incremental streaming)
        if len(full_content) > last_content_length:
            new_content = full_content[last_content_length:]

            # Print model label only on first chunk
            if last_content_length == 0:
                print(f"[{model_name}]: ", end="", flush=True)
                last_model_name = model_name

            print(new_content, end="", flush=True)
            last_content_length = len(full_content)

    # Add final newline after streaming completes
    if last_content_length > 0:
        print()

    return display_hist, logic_hist


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
                "エラー: 無効な名前です。許可される文字は英数字、ハイフン、アンダースコアのみです。"
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
        new_history, new_system_prompt, new_is_dirty = _handle_history_list(user_id, store)
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


def _handle_copy_command(args, history):
    """Handle the '/copy <index>' command to copy LLM responses.

    Args:
        args: コマンド引数文字列（インデックス番号）。
        history: 現在の会話履歴のリスト。
    """
    if not args:
        print("エラー: コピーするLLM応答のインデックスを指定してください。")
        return

    try:
        index = int(args)
    except ValueError:
        print("エラー: インデックスは整数で指定してください。")
        return

    try:
        message = get_llm_response(history, index)
    except IndexError:
        print(f"エラー: 指定されたインデックスのLLM応答が見つかりません (index={index})。")
        return

    try:
        response_text = ""
        if isinstance(message, list):
            # Extract text from the new structured format
            text_parts = [
                part["content"]
                for part in message
                if part.get("type") == "text" and "content" in part
            ]
            response_text = " ".join(text_parts)
        elif isinstance(message, str):
            # Handle legacy string format for backward compatibility
            response_text = message

        pyperclip.copy(response_text)
        print(f"LLM応答をクリップボードにコピーしました (index={index})。")
    except pyperclip.PyperclipException:
        print("エラー: クリップボードへのコピーに失敗しました。")


async def main():
    """Main CLI loop"""
    import asyncio

    from .mcp.client import MCPClient

    store = HistoryStore()
    user_id = _prompt_user_id()

    history = []
    system_prompt = ""
    is_dirty = False

    # Create session-scoped ChatService for provider reuse
    service = ChatService()

    # MCP support
    mcp_enabled = core.MCP_ENABLED
    mcp_client = None

    if mcp_enabled:
        # TODO: Load command/args from config
        mcp_client = MCPClient("uvx", ["mcp-server-weather"])

    async def run_cli():
        nonlocal history, system_prompt, is_dirty

        # Use MCP client if enabled
        if mcp_client:
            async with mcp_client:
                service.mcp_client = mcp_client
                await _cli_loop()
        else:
            await _cli_loop()

    async def _cli_loop():
        nonlocal history, system_prompt, is_dirty
        while True:
            # Use loop.run_in_executor to make input() non-blocking for asyncio if needed
            # but for a simple CLI, sync input is usually fine if we don't have background tasks.
            try:
                loop = asyncio.get_event_loop()
                prompt = await loop.run_in_executor(None, input, "> ")
                prompt = prompt.strip()
            except EOFError:
                break

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
                elif command == "/reset":
                    if is_dirty and not _confirm(
                        "現在の会話は保存されていません。リセットしますか？"
                    ):
                        continue
                    history = reset_history()
                    is_dirty = bool(system_prompt)  # system promptのみの場合はdirty扱いを継続
                    print("チャット履歴をリセットしました。")
                elif command == "/history":
                    history, system_prompt, is_dirty = _handle_history_command(
                        args, user_id, store, history, system_prompt, is_dirty
                    )
                elif command == "/copy":
                    _handle_copy_command(args, history)
                else:
                    print(
                        f"エラー: `{command}` は不明なコマンドです。"
                        f"利用可能なコマンドは `/system` などです。"
                    )
                continue

            # Use ChatService for message processing (Issue #62)
            # ChatService handles mention parsing, LLM routing, and history updates
            # Update service state with current history and system prompt
            service.display_history = []  # CLI doesn't use Gradio-style display history
            service.logic_history = history
            service.system_prompt = system_prompt

            # Process message through ChatService and handle CLI-specific display
            _, history = await _process_service_stream(service, prompt)
            is_dirty = True

    await run_cli()
    return history, system_prompt


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
