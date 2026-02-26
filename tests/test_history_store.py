import asyncio
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
    assert loaded["turns"] == [
        {"role": "user", "content": [{"type": "text", "content": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "content": "there"}]},
    ]
    assert loaded["system_prompt"] == "sys"

    metadata = loaded["metadata"]
    assert metadata["schema_version"] == 2  # Version 2: structured content format
    # ISO 8601 parse check
    datetime.fromisoformat(metadata["created_at"])
    datetime.fromisoformat(metadata["updated_at"])

    # Verify JSON structure on disk
    raw = json.loads(saved_path.read_text())
    assert raw["display_name"] == "My Chat"
    assert raw["metadata"]["schema_version"] == 2  # Version 2: structured content


def test_history_store_list(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    store.save_history("user-1", "B chat", system_prompt="", turns=[])
    store.save_history("user-1", "A chat", system_prompt="", turns=[])

    names = store.list_histories("user-1")
    assert names == ["A chat", "B chat"]


def test_save_and_load_autosave(tmp_path):
    store = HistoryStore(base_dir=tmp_path)

    turns = [{"role": "user", "content": "draft message"}]
    store.save_autosave("user-1", system_prompt="autosave sys", turns=turns)

    assert store.has_autosave("user-1") is True

    loaded = store.load_autosave("user-1")
    assert loaded is not None
    assert loaded["type"] == "autosave_draft"
    assert loaded["schema_version"] == 1
    assert loaded["user_id"] == "user-1"
    assert loaded["system_prompt"] == "autosave sys"
    assert loaded["turns"] == [
        {"role": "user", "content": [{"type": "text", "content": "draft message"}]}
    ]
    datetime.fromisoformat(loaded["metadata"]["updated_at"])


def test_autosave_schema_mismatch_ignored(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    store.save_autosave("user-1", system_prompt="sys", turns=[])

    autosave_path = tmp_path / "user-1" / "_autosave.json"
    raw = json.loads(autosave_path.read_text())
    raw["schema_version"] = 999
    autosave_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))

    # schema mismatch should not raise; it should be ignored
    assert store.load_autosave("user-1") is None


def test_autosave_does_not_break_manual_history(tmp_path):
    store = HistoryStore(base_dir=tmp_path)

    store.save_autosave(
        "user-1",
        system_prompt="autosave sys",
        turns=[{"role": "user", "content": "draft"}],
    )

    store.save_history(
        "user-1",
        "Named Chat",
        system_prompt="manual sys",
        turns=[{"role": "user", "content": "manual"}],
    )

    manual_loaded = store.load_history("user-1", "Named Chat")
    assert manual_loaded["display_name"] == "Named Chat"
    assert manual_loaded["system_prompt"] == "manual sys"
    assert manual_loaded["turns"] == [
        {"role": "user", "content": [{"type": "text", "content": "manual"}]}
    ]

    autosave_loaded = store.load_autosave("user-1")
    assert autosave_loaded is not None
    assert autosave_loaded["system_prompt"] == "autosave sys"


def test_manual_history_reserved_name_rejected_for_save(tmp_path):
    store = HistoryStore(base_dir=tmp_path)

    assert store.history_exists("user-1", "_autosave") is False

    with pytest.raises(ValueError):
        store.save_history("user-1", "_autosave", system_prompt="", turns=[])

    with pytest.raises(FileNotFoundError):
        store.load_history("user-1", "_autosave")


def test_history_exists_returns_true_even_for_invalid_json_file(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    user_dir = tmp_path / "user-1"
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "Broken.json").write_text("{invalid json")

    assert store.history_exists("user-1", "Broken") is True


def test_load_history_supports_legacy_reserved_filename(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    user_dir = tmp_path / "user-1"
    user_dir.mkdir(parents=True, exist_ok=True)

    legacy_payload = {
        "display_name": "_autosave",
        "system_prompt": "legacy",
        "turns": [{"role": "user", "content": "hello"}],
        "metadata": {"schema_version": 1},
    }
    (user_dir / "_autosave.json").write_text(
        json.dumps(legacy_payload, ensure_ascii=False, indent=2)
    )

    loaded = store.load_history("user-1", "_autosave")
    assert loaded["display_name"] == "_autosave"
    assert loaded["system_prompt"] == "legacy"


def test_load_autosave_invalid_json_shape_ignored(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    autosave_path = tmp_path / "user-1" / "_autosave.json"
    autosave_path.parent.mkdir(parents=True, exist_ok=True)

    # Top-level JSON is not an object
    autosave_path.write_text(json.dumps([]))
    assert store.load_autosave("user-1") is None

    # metadata is not an object
    autosave_path.write_text(
        json.dumps(
            {
                "type": "autosave_draft",
                "schema_version": 1,
                "user_id": "user-1",
                "system_prompt": "",
                "turns": [],
                "metadata": "invalid",
            }
        )
    )
    assert store.load_autosave("user-1") is None


def test_list_histories_filters_out_autosave(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    store.save_autosave(
        "user-1",
        system_prompt="autosave sys",
        turns=[{"role": "user", "content": "draft"}],
    )
    store.save_history(
        "user-1",
        "Named Chat",
        system_prompt="manual sys",
        turns=[{"role": "user", "content": "manual"}],
    )

    names = store.list_histories("user-1")
    assert names == ["Named Chat"]


def test_autosave_type_mismatch_ignored(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    store.save_autosave("user-1", system_prompt="sys", turns=[])

    autosave_path = tmp_path / "user-1" / "_autosave.json"
    raw = json.loads(autosave_path.read_text())
    raw["type"] = "wrong_type"
    autosave_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))

    assert store.load_autosave("user-1") is None


def test_has_autosave_false_when_no_autosave_exists(tmp_path):
    store = HistoryStore(base_dir=tmp_path)
    assert store.has_autosave("user-1") is False


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
            history, system_prompt = asyncio.run(cli.main())

    assert system_prompt == ""
    # After load, it should restore the history as it was when saved—containing only "hello"—
    # because the save happened before "more" was entered. "more" is not present after restore.
    assert [entry["content"] for entry in history] == [[{"type": "text", "content": "hello"}]]


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
                history, _ = asyncio.run(cli.main())

    # Overwrite accepted on second prompt; history should contain the latest single message
    assert [entry["content"] for entry in history] == [[{"type": "text", "content": "first"}]]
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
            history, _ = asyncio.run(cli.main())

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
            _, _ = asyncio.run(cli.main())

    assert any("chat1" in str(call) for call in mock_print.call_args_list)


def test_load_legacy_string_content_history(tmp_path):
    """レガシー形式（content: str）の履歴を正しく読み込めること"""
    from multi_llm_chat.history import get_llm_response

    store = HistoryStore(base_dir=tmp_path)

    # レガシー形式で直接保存（normalize前の形式）
    legacy_turns = [
        {"role": "user", "content": "hello"},
        {"role": "gemini", "content": "hi there"},
        {"role": "user", "content": "how are you?"},
        {"role": "chatgpt", "content": "I'm good"},
    ]

    store.save_history("user-1", "Legacy Chat", system_prompt="You are helpful", turns=legacy_turns)

    # 正規化されたデータを取得
    loaded = store.load_history("user-1", "Legacy Chat")

    # 構造化された形式に変換されていること
    assert loaded["turns"][0]["content"] == [{"type": "text", "content": "hello"}]
    assert loaded["turns"][1]["content"] == [{"type": "text", "content": "hi there"}]
    assert loaded["turns"][2]["content"] == [{"type": "text", "content": "how are you?"}]
    assert loaded["turns"][3]["content"] == [{"type": "text", "content": "I'm good"}]

    # get_llm_responseでも正しく取得できること
    response = get_llm_response(loaded["turns"], 0)
    assert response == "I'm good"

    response = get_llm_response(loaded["turns"], 1)
    assert response == "hi there"


def test_save_normalizes_content_format(tmp_path):
    """保存時に構造化形式に正規化されること"""
    store = HistoryStore(base_dir=tmp_path)

    # レガシー形式で保存
    legacy_turns = [
        {"role": "user", "content": "hello"},
        {"role": "gemini", "content": "hi there"},
    ]

    store.save_history("user-1", "Test Chat", system_prompt="You are helpful", turns=legacy_turns)

    # 読み込み（自動的に正規化される）
    loaded = store.load_history("user-1", "Test Chat")

    # 構造化形式に変換されていること
    assert loaded["turns"][0]["content"] == [{"type": "text", "content": "hello"}]
    assert loaded["turns"][1]["content"] == [{"type": "text", "content": "hi there"}]

    # 再保存して再読み込み - 構造化形式が維持されること
    store.save_history(
        "user-1", "Test Chat", system_prompt="You are helpful", turns=loaded["turns"]
    )
    reloaded = store.load_history("user-1", "Test Chat")

    # 構造化形式が維持されていること
    assert reloaded["turns"][0]["content"] == [{"type": "text", "content": "hello"}]
    assert reloaded["turns"][1]["content"] == [{"type": "text", "content": "hi there"}]
