"""Tests for context compression and token guard rail features (Task A)"""

import os
from unittest.mock import patch

import multi_llm_chat.core as core


def test_max_context_length_from_env():
    """Should read model-specific max context length from environment variables"""
    with patch.dict(
        os.environ,
        {
            "GEMINI_MAX_CONTEXT_LENGTH": "100000",
            "CHATGPT_MAX_CONTEXT_LENGTH": "50000",
        },
    ):
        # Reload to pick up env vars
        gemini_result = core.get_max_context_length("gemini-1.5-pro")
        chatgpt_result = core.get_max_context_length("gpt-4o")

        assert gemini_result == 100000
        assert chatgpt_result == 50000


def test_max_context_length_fallback_to_default():
    """Should fall back to DEFAULT_MAX_CONTEXT_LENGTH when model-specific setting is missing"""
    with patch.dict(
        os.environ,
        {"DEFAULT_MAX_CONTEXT_LENGTH": "8192"},
        clear=False,
    ):
        # Model without specific setting should use default
        result = core.get_max_context_length("unknown-model")
        assert result == 8192


def test_max_context_length_uses_built_in_default():
    """Should use built-in default (4096) when no env vars are set"""
    with patch.dict(os.environ, {}, clear=True):
        result = core.get_max_context_length("any-model")
        assert result == 4096


def test_token_calculation_with_buffer_factor():
    """Should apply TOKEN_ESTIMATION_BUFFER_FACTOR to estimated tokens for non-OpenAI models"""
    text = "This is a test message with some content"

    with patch.dict(os.environ, {"TOKEN_ESTIMATION_BUFFER_FACTOR": "1.2"}):
        result = core.calculate_tokens(text, "gemini-1.5-pro")

        # Should apply 1.2x buffer to estimation
        base_estimate = core._estimate_tokens(text)
        expected = int(base_estimate * 1.2)
        assert result == expected


def test_openai_uses_tiktoken():
    """Should use tiktoken for accurate OpenAI token counting"""
    text = "Hello, world! This is a test."

    # For OpenAI models, should use tiktoken (not estimation with buffer)
    result = core.calculate_tokens(text, "gpt-4o")

    # Result should be from tiktoken, not estimation
    # We can verify by checking it's different from buffered estimation
    base_estimate = core._estimate_tokens(text)
    buffered_estimate = int(base_estimate * 1.2)

    # tiktoken result should be different (more accurate)
    assert result != buffered_estimate
    assert result > 0


def test_sliding_window_pruning_basic():
    """Should prune old conversation turns to fit within max context length"""
    # Create history that exceeds limit with longer messages
    history = [
        {"role": "user", "content": "This is the first message with some content " * 10},
        {"role": "gemini", "content": "This is the first response with some content " * 10},
        {"role": "user", "content": "This is the second message with some content " * 10},
        {"role": "gemini", "content": "This is the second response with some content " * 10},
        {"role": "user", "content": "This is the third message with some content " * 10},
        {"role": "gemini", "content": "This is the third response with some content " * 10},
    ]

    # Set a limit to force pruning
    max_tokens = 200

    pruned = core.prune_history_sliding_window(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
    )

    # Should keep only the most recent turns that fit
    assert len(pruned) < len(history)
    # Should keep the most recent message
    assert pruned[-1]["content"] == history[-1]["content"]


def test_sliding_window_preserves_system_prompt():
    """Should always preserve system prompt when pruning history"""
    history = [
        {"role": "user", "content": "Message 1"},
        {"role": "gemini", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "gemini", "content": "Response 2"},
    ]

    system_prompt = "You are a helpful assistant."
    max_tokens = 50  # Very small to force aggressive pruning

    pruned = core.prune_history_sliding_window(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=system_prompt
    )

    # System prompt tokens should be accounted for, but not in the history
    # The most recent turns should remain
    assert len(pruned) >= 2  # At least latest user+assistant pair
    assert pruned[-1]["content"] == "Response 2"


def test_pruning_by_turn_pairs():
    """Should prune complete conversation turns (user + assistant pairs)"""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "gemini", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "gemini", "content": "A2"},
        {"role": "user", "content": "Q3"},
    ]

    max_tokens = 30  # Force removal of some turns

    pruned = core.prune_history_sliding_window(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
    )

    # If Q1-A1 is removed, both should be removed (turn pair)
    if len(pruned) < len(history):
        # Check that we don't have orphaned responses
        for i, entry in enumerate(pruned):
            if entry["role"] in ["gemini", "chatgpt"]:
                # Assistant response should have preceding user message
                assert i > 0
                # Previous entry should be user message
                assert pruned[i - 1]["role"] == "user"


def test_system_prompt_exceeds_limit():
    """Should raise error when system prompt alone exceeds max context length"""
    # Create a very long system prompt
    long_prompt = "A" * 10000  # Very long prompt
    model_name = "gemini-1.5-pro"

    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "1000"}):
        # Should raise an error or return validation result
        result = core.validate_system_prompt_length(long_prompt, model_name)
        assert result["valid"] is False
        assert "exceeds" in result["error"].lower()


def test_single_turn_exceeds_limit():
    """Should detect when system prompt + latest turn exceeds max context"""
    system_prompt = "You are a helpful assistant."
    history = [
        {"role": "user", "content": "A" * 5000},  # Very long user message
    ]

    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "1000"}):
        result = core.validate_context_length(history, system_prompt, model_name="gemini-1.5-pro")

        assert result["valid"] is False
        assert "single turn" in result["error"].lower() or "too long" in result["error"].lower()


def test_context_overflow_warning():
    """Should provide warning info when context would be pruned"""
    history = [
        {"role": "user", "content": "Message " + "x" * 100},
        {"role": "gemini", "content": "Response " + "y" * 100},
        {"role": "user", "content": "Message " + "z" * 100},
        {"role": "gemini", "content": "Response " + "w" * 100},
    ]

    max_tokens = 50  # Force pruning

    info = core.get_pruning_info(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
    )

    assert "turns_to_remove" in info
    assert info["turns_to_remove"] > 0
    assert "original_length" in info
    assert "pruned_length" in info


def test_invalid_env_var_logs_warning():
    """Should log warning when environment variable has invalid value"""
    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "not_a_number"}):
        with patch("logging.warning") as mock_warning:
            result = core.get_max_context_length("gemini-1.5-pro")

            # Should fall back to default
            assert result == 4096
            # Should log warning about invalid value
            mock_warning.assert_called_once()
            assert "GEMINI_MAX_CONTEXT_LENGTH" in str(mock_warning.call_args)
