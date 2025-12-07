from unittest.mock import patch

import pytest

import multi_llm_chat.chat_logic as chat_logic


def _gemini_stream(*_args, **_kwargs):
    return iter(["Mocked Gemini Response"])


def _chatgpt_stream(*_args, **_kwargs):
    return iter(["Mocked ChatGPT Response"])


def test_repl_exit_commands():
    with patch("builtins.input", side_effect=["test-user", "hello", "exit"]):
        with patch("builtins.print"):
            chat_logic.main()
            assert True

    with patch("builtins.input", side_effect=["test-user", "hello", "quit"]):
        with patch("builtins.print"):
            chat_logic.main()
            assert True


def test_history_management_user_input():
    test_inputs = ["hello gemini", "@gemini how are you?", "just a thought"]

    with patch("builtins.input", side_effect=test_inputs + ["exit"]):
        with patch("builtins.print"):  # Mock print to avoid console output
            # Mock API calls to control history length
            with patch("multi_llm_chat.core.call_gemini_api", side_effect=_gemini_stream):
                with patch("multi_llm_chat.core.call_chatgpt_api", side_effect=_chatgpt_stream):
                    history = chat_logic.main()

            # Expected history: user, user, gemini, user
            assert len(history) == 4
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "hello gemini"
            assert history[1]["role"] == "user"
            assert history[1]["content"] == "@gemini how are you?"
            assert history[2]["role"] == "gemini"
            assert history[2]["content"] == "Mocked Gemini Response"
            assert history[3]["role"] == "user"
            assert history[3]["content"] == "just a thought"


# Mock API calls for testing routing and API responses
@patch("multi_llm_chat.core.call_gemini_api", side_effect=_gemini_stream)
@patch("multi_llm_chat.core.call_chatgpt_api", side_effect=_chatgpt_stream)
def test_mention_routing(mock_chatgpt_api, mock_gemini_api):
    # Test @gemini
    with patch("builtins.input", side_effect=["test-user", "@gemini hello", "exit"]):
        with patch("builtins.print"):
            history = chat_logic.main()
            assert mock_gemini_api.called
            assert not mock_chatgpt_api.called
            assert history[-1]["role"] == "gemini"
            assert history[-1]["content"] == "Mocked Gemini Response"
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test @chatgpt
    with patch("builtins.input", side_effect=["test-user", "@chatgpt hello", "exit"]):
        with patch("builtins.print"):
            history = chat_logic.main()
            assert not mock_gemini_api.called
            assert mock_chatgpt_api.called
            assert history[-1]["role"] == "chatgpt"
            assert history[-1]["content"] == "Mocked ChatGPT Response"
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test @all
    with patch("builtins.input", side_effect=["test-user", "@all hello", "exit"]):
        with patch("builtins.print"):
            history_snapshots = {}

            def gemini_capture(history, system_prompt=None):
                history_snapshots["gemini"] = [entry.copy() for entry in history]
                return iter(["Mocked Gemini Response"])

            def chatgpt_capture(history, system_prompt=None):
                history_snapshots["chatgpt"] = [entry.copy() for entry in history]
                return iter(["Mocked ChatGPT Response"])

            mock_gemini_api.side_effect = gemini_capture
            mock_chatgpt_api.side_effect = chatgpt_capture

            history = chat_logic.main()
            assert mock_gemini_api.called
            assert mock_chatgpt_api.called
            # Check the last two entries for @all
            assert history[-2]["role"] == "gemini"
            assert history[-2]["content"] == "Mocked Gemini Response"
            assert history[-1]["role"] == "chatgpt"
            assert history[-1]["content"] == "Mocked ChatGPT Response"
            assert history_snapshots["gemini"] == history_snapshots["chatgpt"]
            assert all(entry["role"] != "gemini" for entry in history_snapshots["chatgpt"])

            mock_gemini_api.side_effect = _gemini_stream
            mock_chatgpt_api.side_effect = _chatgpt_stream
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test no mention
    with patch("builtins.input", side_effect=["test-user", "hello", "exit"]):
        with patch("builtins.print"):
            history = chat_logic.main()
            assert not mock_gemini_api.called
            assert not mock_chatgpt_api.called
            assert history[-1]["role"] == "user"
            assert history[-1]["content"] == "hello"


def test_reset_command_clears_history(monkeypatch):
    """chat_logic.main should honor /reset and return only post-reset messages"""
    monkeypatch.setenv("CHAT_HISTORY_USER_ID", "test-user")
    test_inputs = [
        "/system base prompt",
        "before reset",
        "/reset",
        "y",
        "after reset",
        "exit",
    ]

    with patch("builtins.input", side_effect=test_inputs):
        with patch("builtins.print"):
            history = chat_logic.main()

    assert [entry["content"] for entry in history] == ["after reset"]


def test_get_llm_response_by_index():
    """指定インデックスのLLM応答を取得できること"""
    history = [
        {"role": "user", "content": "hi"},
        {"role": "gemini", "content": "G-1"},
        {"role": "user", "content": "hello"},
        {"role": "chatgpt", "content": "C-1"},
    ]

    assert chat_logic.get_llm_response(history, 0) == "C-1"
    assert chat_logic.get_llm_response(history, 1) == "G-1"


def test_get_llm_response_raises_on_missing():
    """LLM応答が存在しない場合はIndexErrorとなること"""
    history = [{"role": "user", "content": "only user"}]

    with pytest.raises(IndexError):
        chat_logic.get_llm_response(history, 0)
