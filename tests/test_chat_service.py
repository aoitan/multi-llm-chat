"""Tests for ChatService - business logic layer for chat operations"""

import unittest
from unittest.mock import patch

from multi_llm_chat.chat_logic import ChatService


class TestChatServiceBasics(unittest.TestCase):
    """Test basic ChatService initialization and state management"""

    def test_create_service_with_empty_history(self):
        """ChatService should initialize with empty histories"""
        service = ChatService()

        assert service.display_history == []
        assert service.logic_history == []
        assert service.system_prompt == ""

    def test_create_service_with_existing_data(self):
        """ChatService should accept initial state"""
        display_hist = [["user msg", "assistant msg"]]
        logic_hist = [
            {"role": "user", "content": "user msg"},
            {"role": "assistant", "content": "assistant msg"},
        ]
        sys_prompt = "Test prompt"

        service = ChatService(
            display_history=display_hist,
            logic_history=logic_hist,
            system_prompt=sys_prompt,
        )

        assert service.display_history == display_hist
        assert service.logic_history == logic_hist
        assert service.system_prompt == sys_prompt


class TestChatServiceMessageParsing(unittest.TestCase):
    """Test mention parsing logic"""

    def test_parse_mention_gemini(self):
        """Should detect @gemini mention"""
        service = ChatService()
        mention = service.parse_mention("@gemini tell me about Python")

        assert mention == "gemini"

    def test_parse_mention_chatgpt(self):
        """Should detect @chatgpt mention"""
        service = ChatService()
        mention = service.parse_mention("@chatgpt explain async")

        assert mention == "chatgpt"

    def test_parse_mention_all(self):
        """Should detect @all mention"""
        service = ChatService()
        mention = service.parse_mention("@all compare these two")

        assert mention == "all"

    def test_parse_mention_none(self):
        """Should return None for messages without mentions"""
        service = ChatService()
        mention = service.parse_mention("regular message")

        assert mention is None

    def test_parse_mention_ignores_whitespace(self):
        """Should handle leading/trailing whitespace"""
        service = ChatService()
        mention = service.parse_mention("  @gemini  ")

        assert mention == "gemini"


class TestChatServiceProcessMessage(unittest.TestCase):
    """Test main message processing logic"""

    @patch("multi_llm_chat.chat_logic.call_gemini_api")
    def test_process_message_gemini(self, mock_gemini):
        """Should call Gemini API for @gemini mention"""
        mock_gemini.return_value = iter(["Test ", "response"])

        service = ChatService()
        results = list(service.process_message("@gemini hello"))

        # Should have yielded at least once
        assert len(results) > 0

        # Final state should include user message and response
        final_display, final_logic = results[-1]
        assert len(final_logic) == 2  # user + assistant
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "gemini"
        assert "Test response" in final_logic[1]["content"]

    @patch("multi_llm_chat.chat_logic.call_chatgpt_api")
    def test_process_message_chatgpt(self, mock_chatgpt):
        """Should call ChatGPT API for @chatgpt mention"""
        mock_chatgpt.return_value = iter(["Hello ", "world"])

        service = ChatService()
        results = list(service.process_message("@chatgpt hi"))

        final_display, final_logic = results[-1]
        assert len(final_logic) == 2
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "chatgpt"
        assert "Hello world" in final_logic[1]["content"]

    @patch("multi_llm_chat.chat_logic.call_gemini_api")
    @patch("multi_llm_chat.chat_logic.call_chatgpt_api")
    def test_process_message_all(self, mock_chatgpt, mock_gemini):
        """Should call both APIs for @all mention"""
        mock_gemini.return_value = iter(["Gemini response"])
        mock_chatgpt.return_value = iter(["ChatGPT response"])

        service = ChatService()
        results = list(service.process_message("@all compare"))

        final_display, final_logic = results[-1]
        # Should have user message + 2 responses
        assert len(final_logic) == 3
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[2]["role"] == "chatgpt"

    def test_process_message_no_mention_raises_error(self):
        """Should raise error for messages without mention"""
        service = ChatService()

        with self.assertRaises(ValueError):
            list(service.process_message("no mention here"))


class TestChatServiceHistorySnapshot(unittest.TestCase):
    """Test history snapshot logic for @all"""

    @patch("multi_llm_chat.chat_logic.call_gemini_api")
    @patch("multi_llm_chat.chat_logic.call_chatgpt_api")
    def test_all_uses_same_history_snapshot(self, mock_chatgpt, mock_gemini):
        """@all should use identical history for both LLMs"""
        captured_histories = []

        def capture_gemini(history, system_prompt):
            captured_histories.append(("gemini", [h.copy() for h in history]))
            return iter(["Gemini"])

        def capture_chatgpt(history, system_prompt):
            captured_histories.append(("chatgpt", [h.copy() for h in history]))
            return iter(["ChatGPT"])

        mock_gemini.side_effect = capture_gemini
        mock_chatgpt.side_effect = capture_chatgpt

        service = ChatService()
        list(service.process_message("@all test"))

        # Both should have received same history (user message only)
        assert len(captured_histories) == 2
        gemini_hist = captured_histories[0][1]
        chatgpt_hist = captured_histories[1][1]

        # Both should be identical and contain only user message
        assert gemini_hist == chatgpt_hist
        assert len(gemini_hist) == 1
        assert gemini_hist[0]["role"] == "user"


class TestChatServiceSystemPrompt(unittest.TestCase):
    """Test system prompt handling"""

    @patch("multi_llm_chat.chat_logic.call_gemini_api")
    def test_system_prompt_passed_to_api(self, mock_gemini):
        """System prompt should be passed to LLM API"""
        mock_gemini.return_value = iter(["Response"])

        service = ChatService(system_prompt="You are a helpful assistant")
        list(service.process_message("@gemini hello"))

        # Check that system prompt was passed (2nd positional arg)
        call_args = mock_gemini.call_args
        assert call_args[0][1] == "You are a helpful assistant"

    def test_update_system_prompt(self):
        """Should allow updating system prompt"""
        service = ChatService()
        service.set_system_prompt("New prompt")

        assert service.system_prompt == "New prompt"


if __name__ == "__main__":
    unittest.main()
