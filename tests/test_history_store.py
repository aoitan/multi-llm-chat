import json
from datetime import datetime
from unittest.mock import patch

import pytest

from multi_llm_chat import cli
from multi_llm_chat.history import HistoryStore, sanitize_name


def test_sanitize_name_rules():
    assert sanitize_name("User-01_ok") == "User-01_ok"
    assert sanitize_name("name with spaces?") == "name_with_spaces_"

    for invalid in ("", ".", ".."):
        with pytest.raises(ValueError):
            sanitize_name(invalid)


def test_history_store_save_load_roundtrip(tmp_path):
    store = HistoryStore(base_dir=tmp_path)

    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}]
    store.save_history("user-1", "My Chat", system_prompt="sys", turns=turns)

    saved_path = tmp_path / "user-1" / "My_Chat.json"
    assert saved_path.exists()

    loaded = store.load_history("user-1", "My Chat")
    assert loaded["display_name"] == "My Chat"
    assert loaded["turns"] == turns
    assert loaded["system_prompt"] == "sys"

    metadata = loaded["metadata"]
    assert metadata["schema_version"] == 1
    # ISO 8601 parse check
    datetime.fromisoformat(metadata["created_at"])
    datetime.fromisoformat(metadata["updated_at"])

    # Verify JSON structure on disk
    raw = json.loads(saved_path.read_text())
    assert raw["display_name"] == "My Chat"
    assert raw["metadata"]["schema_version"] == 1


def test_history_store_list(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    store.save_history("user-1", "B chat", system_prompt="", turns=[])
    store.save_history("user-1", "A chat", system_prompt="", turns=[])

    names = store.list_histories("user-1")
    assert names == ["A chat", "B chat"]


def test_cli_history_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "cli-user")

    inputs = [
        "hello",
        "/history save First",
        "more",
        "/history new",
        "y",
        "/history load First",
        "exit",
    ]

    with patch("builtins.input", side_effect=inputs):
        with patch("builtins.print"):
            history, system_prompt = cli.main()

    assert system_prompt == ""
    # After load, it should restore the saved history (only "hello" and Gemini call was never made)
    assert [entry["content"] for entry in history] == ["hello"]


def test_cli_history_overwrite_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "cli-user")

    inputs = [
        "first",
        "/history save name1",
        "/history save name1",
        "/history save name1",
        "/history load name1",
        "exit",
    ]

    prompts_seen = []

    def fake_confirm(prompt):
        prompts_seen.append(prompt)
        # First overwrite -> decline, second -> accept
        return len(prompts_seen) == 2

    with patch("multi_llm_chat.cli._confirm", side_effect=fake_confirm):
        with patch("builtins.input", side_effect=inputs):
            with patch("builtins.print"):
                history, _ = cli.main()

    # Overwrite accepted on second prompt; history should contain the latest single message
    assert [entry["content"] for entry in history] == ["first"]
    assert any("上書きしますか" in prompt for prompt in prompts_seen)


def test_cli_history_invalid_name_and_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "cli-user")

    inputs = [
        "/history save ../bad",
        "/history load ../bad",
        "/history load missing",
        "exit",
    ]

    with patch("builtins.input", side_effect=inputs):
        with patch("builtins.print") as mock_print:
            history, _ = cli.main()

    # No history should be saved or loaded
    assert history == []
    assert any("無効な名前" in str(call) for call in mock_print.call_args_list)
    assert any("見つかりません" in str(call) for call in mock_print.call_args_list)


def test_cli_history_list_outputs_saved_names(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_DIR", str(tmp_path))

    inputs = [
        "cli-user",
        "first",
        "/history save chat1",
        "/history list",
        "exit",
    ]

    with patch("builtins.input", side_effect=inputs):
        with patch("builtins.print") as mock_print:
            _, _ = cli.main()

    assert any("chat1" in str(call) for call in mock_print.call_args_list)
