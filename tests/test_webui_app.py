
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

    def test_update_ui_on_user_id_change_disables_send_when_tokens_exceeded(
        self, mock_state
    ):
        """
        ユーザーID変更時、トークン数が上限を超える場合に送信ボタンが無効化されるかをテストする
        """
        # 準備: 長いシステムプロンプトと履歴でトークン上限を確実に超えるようにする
        long_system_prompt = "This is a very long system prompt... " * 500
        # `update_ui_on_user_id_change` は内部で `WebUIState` を呼び出す
        # この関数を直接テストすることで、GradioのUIコンポーネントの状態更新を検証する
        initial_history_str = logic_history_to_display(
            mock_state["logic_history"]
        )

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

    def test_update_ui_on_user_id_change_enables_buttons_with_valid_user_id(
        self, mock_state
    ):
        """
        有効なユーザーIDが入力された場合、ボタンが有効になるかをテストする
        """
        initial_history_str = logic_history_to_display(
            mock_state["logic_history"]
        )
        with patch(
            "multi_llm_chat.webui.state.core.get_token_info",
            return_value={"token_count": 100, "max_context_length": 8192},
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

    def test_update_ui_on_user_id_change_disables_buttons_with_empty_user_id(
        self, mock_state
    ):
        """
        ユーザーIDが空の場合、ボタンが無効になるかをテストする
        """
        initial_history_str = logic_history_to_display(
            mock_state["logic_history"]
        )
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
