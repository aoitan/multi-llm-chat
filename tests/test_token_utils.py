import pytest
from multi_llm_chat.token_utils import estimate_tokens, get_max_context_length, get_buffer_factor

def test_estimate_tokens():
    # ASCII text: ~4 chars = 1 token
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    
    # Japanese text: ~1.5 chars = 1 token
    assert estimate_tokens("あいう") == 2
    assert estimate_tokens("あいうえお") == 3
    
    # Mixed
    # "Hello" (5 chars) -> 5/4 = 1.25
    # "こんにちは" (5 chars) -> 5/1.5 = 3.33
    # Total = 4.58 -> 4 (int conversion)
    assert estimate_tokens("Helloこんにちは") == 4

def test_get_max_context_length(monkeypatch):
    # Clear environment variables to test defaults
    monkeypatch.delenv("GEMINI_MAX_CONTEXT_LENGTH", raising=False)
    monkeypatch.delenv("CHATGPT_MAX_CONTEXT_LENGTH", raising=False)
    monkeypatch.delenv("DEFAULT_MAX_CONTEXT_LENGTH", raising=False)
    
    # Default for gemini
    assert get_max_context_length("gemini-pro") == 32760
    
    # Environment variable override
    monkeypatch.setenv("GEMINI_MAX_CONTEXT_LENGTH", "1000")
    assert get_max_context_length("gemini-pro") == 1000
    
    # Default for unknown
    monkeypatch.delenv("GEMINI_MAX_CONTEXT_LENGTH", raising=False)
    assert get_max_context_length("unknown-model") == 4096

def test_get_buffer_factor(monkeypatch):
    # Default
    assert get_buffer_factor() == 1.2
    
    # Environment variable override
    monkeypatch.setenv("TOKEN_ESTIMATION_BUFFER_FACTOR", "1.5")
    assert get_buffer_factor() == 1.5
