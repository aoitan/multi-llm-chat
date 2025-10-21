import pytest
from unittest.mock import patch
import main

def test_repl_exit_commands():
    with patch('builtins.input', side_effect=["hello", "exit"]):
        with patch('builtins.print') as mock_print:
            main.main()
            # Ensure exit command terminates the loop
            assert True # If it reaches here without error, it exited

    with patch('builtins.input', side_effect=["hello", "quit"]):
        with patch('builtins.print') as mock_print:
            main.main()
            # Ensure quit command terminates the loop
            assert True # If it reaches here without error, it exited

def test_history_management_user_input():
    test_inputs = ["hello gemini", "@gemini how are you?", "just a thought"]
    
    with patch('builtins.input', side_effect=test_inputs + ["exit"]):
        with patch('builtins.print'): # Mock print to avoid console output
            # Mock API calls to control history length
            with patch('main.call_gemini_api', return_value="Mocked Gemini Response"):
                with patch('main.call_chatgpt_api', return_value="Mocked ChatGPT Response"):
                    history = main.main()
            
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
@patch('main.call_gemini_api', return_value="Mocked Gemini Response")
@patch('main.call_chatgpt_api', return_value="Mocked ChatGPT Response")
def test_mention_routing(mock_chatgpt_api, mock_gemini_api):
    # Test @gemini
    with patch('builtins.input', side_effect=["@gemini hello", "exit"]):
        with patch('builtins.print'):
            history = main.main()
            assert mock_gemini_api.called
            assert not mock_chatgpt_api.called
            assert history[-1]["role"] == "gemini"
            assert history[-1]["content"] == "Mocked Gemini Response"
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test @chatgpt
    with patch('builtins.input', side_effect=["@chatgpt hello", "exit"]):
        with patch('builtins.print'):
            history = main.main()
            assert not mock_gemini_api.called
            assert mock_chatgpt_api.called
            assert history[-1]["role"] == "chatgpt"
            assert history[-1]["content"] == "Mocked ChatGPT Response"
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test @all
    with patch('builtins.input', side_effect=["@all hello", "exit"]):
        with patch('builtins.print'):
            history = main.main()
            assert mock_gemini_api.called
            assert mock_chatgpt_api.called
            # Check the last two entries for @all
            assert history[-2]["role"] == "gemini"
            assert history[-2]["content"] == "Mocked Gemini Response"
            assert history[-1]["role"] == "chatgpt"
            assert history[-1]["content"] == "Mocked ChatGPT Response"
            mock_gemini_api.reset_mock()
            mock_chatgpt_api.reset_mock()

    # Test no mention
    with patch('builtins.input', side_effect=["hello", "exit"]):
        with patch('builtins.print'):
            history = main.main()
            assert not mock_gemini_api.called
            assert not mock_chatgpt_api.called
            assert history[-1]["role"] == "user"
            assert history[-1]["content"] == "hello"
