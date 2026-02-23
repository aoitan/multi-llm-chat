from unittest.mock import patch

import pytest

from multi_llm_chat.webui.app import (
    update_ui_on_user_id_change,
)
from multi_llm_chat.webui.handlers import logic_history_to_display

INITIAL_SYSTEM_PROMPT = ""


class TestWebUIApp:
    @pytest.fixture
    def mock_state(self):
        """logic_historyを含むstateのモックを返す"""
        state = {
            "logic_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        return state

    def test_update_ui_on_user_id_change_disables_send_when_tokens_exceeded(self, mock_state):
        """
        ユーザーID変更時、トークン数が上限を超える場合に送信ボタンが無効化されるかをテストする
        """
        # 準備: 長いシステムプロンプトと履歴でトークン上限を確実に超えるようにする
        long_system_prompt = "This is a very long system prompt... " * 500
        # `update_ui_on_user_id_change` は内部で `WebUIState` を呼び出す
        # この関数を直接テストすることで、GradioのUIコンポーネントの状態更新を検証する
        initial_history_str = logic_history_to_display(mock_state["logic_history"])

        with patch(
            "multi_llm_chat.webui.state.core.get_token_info",
            return_value={"token_count": 9001, "max_context_length": 8192},
        ):
            updates = update_ui_on_user_id_change(
                "test_user", mock_state["logic_history"], long_system_prompt
            )

        # 検証
        updated_history_str = updates[0]
        system_prompt_output = updates[2]
        # send_buttonはoutputsリストの7番目 (インデックス6)
        send_button_state = updates[6]

        # 履歴とシステムプロンプトが正しく更新されていることを確認
        assert updated_history_str == initial_history_str
        assert system_prompt_output == long_system_prompt

        # 送信ボタンが無効化されていることを確認 (回帰バグの検出)
        assert send_button_state["interactive"] is False

    def test_update_ui_on_user_id_change_enables_buttons_with_valid_user_id(self, mock_state):
        """
        有効なユーザーIDが入力された場合、ボタンが有効になるかをテストする
        """
        initial_history_str = logic_history_to_display(mock_state["logic_history"])
        with (
            patch(
                "multi_llm_chat.webui.state.core.get_token_info",
                return_value={"token_count": 100, "max_context_length": 8192},
            ),
            patch(
                "multi_llm_chat.webui.app.has_history_for_user",
                return_value=True,
            ),
        ):
            updates = update_ui_on_user_id_change(
                "test_user", mock_state["logic_history"], INITIAL_SYSTEM_PROMPT
            )

        updated_history_str = updates[0]
        system_prompt_output = updates[2]
        # ボタンはインデックス 3 から 7
        button_states = updates[3:8]

        assert updated_history_str == initial_history_str
        assert system_prompt_output == INITIAL_SYSTEM_PROMPT
        # すべてのボタンが有効であるべき
        assert all(btn["interactive"] for btn in button_states)

    def test_update_ui_on_user_id_change_disables_buttons_with_empty_user_id(self, mock_state):
        """
        ユーザーIDが空の場合、ボタンが無効になるかをテストする
        """
        initial_history_str = logic_history_to_display(mock_state["logic_history"])
        updates = update_ui_on_user_id_change(
            "", mock_state["logic_history"], INITIAL_SYSTEM_PROMPT
        )

        updated_history_str = updates[0]
        system_prompt_output = updates[2]
        # ボタンはインデックス 3 から 7
        button_states = updates[3:8]

        assert updated_history_str == initial_history_str
        assert system_prompt_output == INITIAL_SYSTEM_PROMPT
        # すべてのボタンが無効であるべき
        assert not any(btn["interactive"] for btn in button_states)

    def test_logic_history_to_display_handles_structured_content(self):
        logic_history = [
            {"role": "user", "content": "hi"},
            {
                "role": "gemini",
                "content": [
                    {"type": "text", "content": "G-1"},
                    {"type": "tool_call", "content": {"name": "tool", "arguments": {}}},
                    {"type": "text", "content": "G-2"},
                ],
            },
        ]

        display_history = logic_history_to_display(logic_history)

        # Now includes tool call in formatted output
        assert display_history == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "**Gemini:**\nG-1 G-2\n\n🔧 **Tool Call**: tool\n"},
        ]

    def test_logic_history_to_display_preserves_tool_execution_logs(self):
        """Test that tool role entries and tool_result are preserved in display history."""
        logic_history = [
            {"role": "user", "content": "search for python"},
            {
                "role": "chatgpt",
                "content": [
                    {"type": "text", "content": "Let me search."},
                    {
                        "type": "tool_call",
                        "content": {
                            "name": "web_search",
                            "arguments": {"query": "python"},
                            "tool_call_id": "call_123",
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_123",
                        "name": "web_search",
                        "content": "Python is a programming language...",
                    }
                ],
            },
            {
                "role": "chatgpt",
                "content": [{"type": "text", "content": "Found results!"}],
            },
        ]

        display_history = logic_history_to_display(logic_history)

        # Tool call and tool result should be visible
        assert len(display_history) == 2
        user_msg = display_history[0]["content"]
        assistant_msg = display_history[1]["content"]
        assert user_msg == "search for python"
        assert "Let me search." in assistant_msg
        assert "🔧 **Tool Call**: web_search" in assistant_msg
        assert "✅ **Result** (web_search):" in assistant_msg
        assert "Found results!" in assistant_msg

    def test_logic_history_to_display_all_produces_separate_bubbles(self):
        """@all の Gemini/ChatGPT 応答はそれぞれ別のアシスタントバブルになること"""
        logic_history = [
            {"role": "user", "content": "@all compare"},
            {"role": "gemini", "content": [{"type": "text", "content": "Gemini answer"}]},
            {"role": "chatgpt", "content": [{"type": "text", "content": "ChatGPT answer"}]},
        ]

        display_history = logic_history_to_display(logic_history)

        # user + gemini bubble + chatgpt bubble = 3 entries
        assert len(display_history) == 3
        assert display_history[0] == {"role": "user", "content": "@all compare"}
        assert display_history[1]["role"] == "assistant"
        assert "**Gemini:**" in display_history[1]["content"]
        assert "Gemini answer" in display_history[1]["content"]
        assert display_history[2]["role"] == "assistant"
        assert "**ChatGPT:**" in display_history[2]["content"]
        assert "ChatGPT answer" in display_history[2]["content"]
