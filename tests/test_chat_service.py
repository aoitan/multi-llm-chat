"""Tests for ChatService - business logic layer for chat operations"""

import unittest
from unittest.mock import MagicMock, patch

from multi_llm_chat.chat_logic import ChatService, parse_mention


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
        mention = parse_mention("@gemini tell me about Python")
        assert mention == "gemini"

    def test_parse_mention_chatgpt(self):
        """Should detect @chatgpt mention"""
        mention = parse_mention("@chatgpt explain async")
        assert mention == "chatgpt"

    def test_parse_mention_all(self):
        """Should detect @all mention"""
        mention = parse_mention("@all compare these two")
        assert mention == "all"

    def test_parse_mention_none(self):
        """Should return None for messages without mentions"""
        mention = parse_mention("regular message")
        assert mention is None

    def test_parse_mention_ignores_whitespace(self):
        """Should handle leading/trailing whitespace"""
        mention = parse_mention("  @gemini  ")
        assert mention == "gemini"


class TestChatServiceProcessMessage(unittest.TestCase):
    """Test main message processing logic"""

    @patch("multi_llm_chat.chat_logic.get_provider")
    def test_process_message_gemini(self, mock_get_provider):
        """Should call Gemini API for @gemini mention"""
        # Setup mock provider
        mock_provider = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Test "
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "response"
        mock_provider.call_api.return_value = iter([mock_chunk1, mock_chunk2])
        mock_provider.extract_text_from_chunk.side_effect = lambda chunk: chunk.text
        mock_get_provider.return_value = mock_provider

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

    @patch("multi_llm_chat.chat_logic.get_provider")
    def test_process_message_chatgpt(self, mock_get_provider):
        """Should call ChatGPT API for @chatgpt mention"""
        # Setup mock provider
        mock_provider = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello "
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = "world"
        mock_provider.call_api.return_value = iter([mock_chunk1, mock_chunk2])
        mock_provider.extract_text_from_chunk.side_effect = ["Hello ", "world"]
        mock_get_provider.return_value = mock_provider

        service = ChatService()
        results = list(service.process_message("@chatgpt hi"))

        final_display, final_logic = results[-1]
        assert len(final_logic) == 2
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "chatgpt"
        assert "Hello world" in final_logic[1]["content"]

    @patch("multi_llm_chat.chat_logic.get_provider")
    def test_process_message_all(self, mock_get_provider):
        """Should call both APIs for @all mention"""
        # Setup mock providers for both calls
        mock_gemini_provider = MagicMock()
        mock_gemini_chunk = MagicMock()
        mock_gemini_chunk.text = "Gemini response"
        mock_gemini_provider.call_api.return_value = iter([mock_gemini_chunk])
        mock_gemini_provider.extract_text_from_chunk.return_value = "Gemini response"

        mock_chatgpt_provider = MagicMock()
        mock_chatgpt_chunk = MagicMock()
        mock_chatgpt_chunk.choices = [MagicMock()]
        mock_chatgpt_chunk.choices[0].delta.content = "ChatGPT response"
        mock_chatgpt_provider.call_api.return_value = iter([mock_chatgpt_chunk])
        mock_chatgpt_provider.extract_text_from_chunk.return_value = "ChatGPT response"

        # Return different providers for gemini and chatgpt
        mock_get_provider.side_effect = [mock_gemini_provider, mock_chatgpt_provider]

        service = ChatService()
        results = list(service.process_message("@all compare"))

        final_display, final_logic = results[-1]
        # Should have user message + 2 responses
        assert len(final_logic) == 3
        assert final_logic[0]["role"] == "user"
        assert final_logic[1]["role"] == "gemini"
        assert final_logic[2]["role"] == "chatgpt"

    def test_process_message_no_mention_as_memo(self):
        """Messages without mention should be added to history as memo (no LLM call)"""
        service = ChatService()

        results = list(service.process_message("This is a memo"))

        # Should yield once (user message added to history)
        assert len(results) == 1
        final_display, final_logic = results[0]

        # User message should be in history
        assert len(final_logic) == 1
        assert final_logic[0]["role"] == "user"
        assert final_logic[0]["content"] == "This is a memo"

        # Display should show user message with no response
        assert len(final_display) == 1
        assert final_display[0][0] == "This is a memo"
        assert final_display[0][1] is None


class TestChatServiceHistorySnapshot(unittest.TestCase):
    """Test history snapshot logic for @all"""

    @patch("multi_llm_chat.chat_logic.get_provider")
    def test_all_uses_same_history_snapshot(self, mock_get_provider):
        """@all should use identical history for both LLMs"""
        captured_histories = []

        def create_mock_provider(provider_name):
            mock_provider = MagicMock()
            
            def capture_call_api(history, system_prompt=None):
                captured_histories.append((provider_name, [h.copy() for h in history]))
                mock_chunk = MagicMock()
                if provider_name == "gemini":
                    mock_chunk.text = "Gemini"
                    mock_provider.extract_text_from_chunk.return_value = "Gemini"
                else:
                    mock_chunk.choices = [MagicMock()]
                    mock_chunk.choices[0].delta.content = "ChatGPT"
                    mock_provider.extract_text_from_chunk.return_value = "ChatGPT"
                return iter([mock_chunk])
            
            mock_provider.call_api.side_effect = capture_call_api
            return mock_provider

        # Return different providers for each call
        mock_get_provider.side_effect = [
            create_mock_provider("gemini"),
            create_mock_provider("chatgpt")
        ]

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

    @patch("multi_llm_chat.chat_logic.get_provider")
    def test_system_prompt_passed_to_api(self, mock_get_provider):
        """System prompt should be passed to LLM API"""
        mock_provider = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Response"
        mock_provider.call_api.return_value = iter([mock_chunk])
        mock_provider.extract_text_from_chunk.return_value = "Response"
        mock_get_provider.return_value = mock_provider

        service = ChatService(system_prompt="You are a helpful assistant")
        list(service.process_message("@gemini hello"))

        # Check that system prompt was passed (2nd positional argument)
        call_args = mock_provider.call_api.call_args
        assert call_args[0][1] == "You are a helpful assistant"

    def test_update_system_prompt(self):
        """Should allow updating system prompt"""
        service = ChatService()
        service.set_system_prompt("New prompt")

        assert service.system_prompt == "New prompt"


if __name__ == "__main__":
    unittest.main()
