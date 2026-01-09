import pytest
from multi_llm_chat.history_utils import get_provider_name_from_model, prepare_request


def test_get_provider_name_from_model():
    assert get_provider_name_from_model("gpt-3.5-turbo") == "chatgpt"
    assert get_provider_name_from_model("GPT-4") == "chatgpt"
    assert get_provider_name_from_model("gemini-pro") == "gemini"
    assert get_provider_name_from_model("GEMINI-1.5-FLASH") == "gemini"


def test_prepare_request_gemini():
    history = [{"role": "user", "content": "hello"}]
    system_prompt = "You are a helpful assistant."

    # Gemini returns tuple (system_prompt, history)
    result = prepare_request(history, system_prompt, "gemini-pro")
    assert result == (system_prompt, history)

    # Empty system prompt
    result = prepare_request(history, "", "gemini-pro")
    assert result == (None, history)


def test_prepare_request_chatgpt():
    history = [{"role": "user", "content": "hello"}]
    system_prompt = "You are a helpful assistant."

    # ChatGPT prepends system prompt to history
    result = prepare_request(history, system_prompt, "gpt-3.5-turbo")
    assert result == [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "hello"},
    ]

    # Empty system prompt
    result = prepare_request(history, "", "gpt-3.5-turbo")
    assert result == history
