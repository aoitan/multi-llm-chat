"""Tests for core.py - Token & Context utilities

This module tests the token counting and context management functions
exposed by the core.py module, specifically focusing on token estimation
and context length retrieval.
"""

import os
from unittest.mock import patch

import multi_llm_chat.core as core


def test_get_token_info_returns_proper_structure():
    """get_token_info should return token count, max context length, and estimation flag"""
    result = core.get_token_info("Hello, world!", "gemini-2.0-flash-exp")
    assert "token_count" in result
    assert "max_context_length" in result
    assert "is_estimated" in result
    assert isinstance(result["token_count"], int)
    assert isinstance(result["max_context_length"], int)
    assert isinstance(result["is_estimated"], bool)


def test_get_token_info_gemini_model():
    """get_token_info should return correct max context for Gemini models"""
    with patch.dict(os.environ, {}, clear=True):
        # Gemini 2.0 Flash (1M)
        result = core.get_token_info("test", "gemini-2.0-flash-exp")
        assert result["max_context_length"] == 1048576

        # Gemini 1.5 Pro (2M)
        result = core.get_token_info("test", "gemini-1.5-pro")
        assert result["max_context_length"] == 2097152

        # Gemini 1.5 Flash (1M)
        result = core.get_token_info("test", "gemini-1.5-flash")
        assert result["max_context_length"] == 1048576

        # Gemini Pro (32K)
        result = core.get_token_info("test", "models/gemini-pro-latest")
        assert result["max_context_length"] == 32760

        # Unknown Gemini variant (conservative default)
        result = core.get_token_info("test", "gemini-unknown")
        assert result["max_context_length"] == 32760


def test_get_token_info_chatgpt_model():
    """get_token_info should return correct max context for ChatGPT models"""
    with patch.dict(os.environ, {}, clear=True):
        result = core.get_token_info("test", "gpt-4o")
        assert result["max_context_length"] == 128000


def test_estimate_tokens_english():
    """Token estimation should handle English text"""
    from multi_llm_chat.core_modules.token_and_context import _estimate_tokens

    # "Hello world" = 11 chars / 4 ≈ 2.75 → 2 tokens
    result = _estimate_tokens("Hello world")
    assert result == 2


def test_estimate_tokens_japanese():
    """Token estimation should handle Japanese text more accurately"""
    from multi_llm_chat.core_modules.token_and_context import _estimate_tokens

    # "こんにちは" = 5 chars / 1.5 ≈ 3.33 → 3 tokens
    result = _estimate_tokens("こんにちは")
    assert result >= 3

    # "日本語テスト" = 6 chars / 1.5 ≈ 4 → 4 tokens
    result = _estimate_tokens("日本語テスト")
    assert result >= 4


def test_estimate_tokens_mixed():
    """Token estimation should handle mixed English/Japanese text"""
    from multi_llm_chat.core_modules.token_and_context import _estimate_tokens

    # "Hello こんにちは" = 5 ASCII + 5 Japanese
    # = (5/4) + (5/1.5) ≈ 1.25 + 3.33 ≈ 4.58 → 4 tokens
    result = _estimate_tokens("Hello こんにちは")
    assert result >= 4
