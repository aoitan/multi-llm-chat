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

    # Set a limit to force pruning (allow 1-2 turns)
    max_tokens = 500

    pruned = core.prune_history_sliding_window(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
    )

    # Should keep only the most recent turns that fit
    assert len(pruned) < len(history)
    assert len(pruned) >= 2  # At least one complete turn
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

            # Should fall back to model default (not 4096)
            assert result == 2097152  # gemini-1.5-pro default
            # Should log warning about invalid value
            mock_warning.assert_called_once()
            assert "GEMINI_MAX_CONTEXT_LENGTH" in str(mock_warning.call_args)


def test_get_token_info_uses_env_based_max_context():
    """get_token_info should use environment-based max context length"""
    with patch.dict(os.environ, {"GEMINI_MAX_CONTEXT_LENGTH": "50000"}):
        result = core.get_token_info("test prompt", "gemini-1.5-pro")

        # Should use env-based max context, not built-in model default
        assert result["max_context_length"] == 50000


def test_pruning_preserves_turn_pairs():
    """Pruning should preserve user-assistant turn pairs"""
    # Create history where pruning would split a turn
    history = [
        {"role": "user", "content": "Q1 " * 50},
        {"role": "gemini", "content": "A1 " * 50},
        {"role": "user", "content": "Q2 " * 50},
        {"role": "gemini", "content": "A2 " * 50},
    ]

    # Set limit that would include A2 but exclude Q2 if we don't preserve pairs
    max_tokens = 150

    pruned = core.prune_history_sliding_window(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
    )

    # Should not have orphaned assistant messages
    for i, entry in enumerate(pruned):
        if entry["role"] in ["gemini", "chatgpt"]:
            # Assistant message must have preceding user message
            assert i > 0, "Assistant message at start (orphaned)"
            assert pruned[i - 1]["role"] == "user", "Assistant without preceding user message"


def test_pruning_removes_orphaned_assistant_messages():
    """Pruning should remove assistant messages if their user message doesn't fit"""
    history = [
        {"role": "user", "content": "Q1 " * 100},  # Large user message
        {"role": "gemini", "content": "A1"},  # Small response
        {"role": "user", "content": "Q2"},
        {"role": "gemini", "content": "A2"},
    ]

    # Very small limit - only latest turn should fit
    max_tokens = 20

    pruned = core.prune_history_sliding_window(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
    )

    # Should only have latest turn (Q2 + A2)
    assert len(pruned) == 2
    assert pruned[0]["content"] == "Q2"
    assert pruned[1]["content"] == "A2"
    # A1 should be removed because Q1 doesn't fit


def test_get_token_info_uses_tiktoken_for_openai():
    """get_token_info should use tiktoken for OpenAI models (accurate counting)"""
    text = "Hello, world! This is a test message."

    result = core.get_token_info(text, "gpt-4o")

    # Should use tiktoken, not estimation
    assert result["token_count"] > 0
    # is_estimated should be False for OpenAI with tiktoken
    assert result["is_estimated"] is False


def test_get_token_info_applies_buffer_for_gemini():
    """get_token_info should apply buffer factor for non-OpenAI models"""
    text = "テストメッセージです"

    with patch.dict(os.environ, {"TOKEN_ESTIMATION_BUFFER_FACTOR": "1.3"}):
        result = core.get_token_info(text, "gemini-1.5-pro")

        # Should apply buffer factor to estimation
        base_estimate = core._estimate_tokens(text)
        expected = int(base_estimate * 1.3)
        assert result["token_count"] == expected


def test_calculate_tokens_invalid_buffer_factor():
    """calculate_tokens should handle invalid buffer factor gracefully"""
    text = "Test message"

    with patch.dict(os.environ, {"TOKEN_ESTIMATION_BUFFER_FACTOR": "invalid"}):
        with patch("logging.warning") as mock_warning:
            # Should not crash, should use default 1.2
            result = core.calculate_tokens(text, "gemini-1.5-pro")

            assert result > 0
            # Should log warning
            mock_warning.assert_called_once()
            assert "TOKEN_ESTIMATION_BUFFER_FACTOR" in str(mock_warning.call_args)


def test_chatgpt_token_count_includes_message_overhead():
    """ChatGPT token counting should include message formatting overhead"""
    # OpenAI's official guidance: each message has ~3 tokens overhead
    # (role name, separators, etc.)
    text = "Hello"  # Simple text

    # Get raw tiktoken count (content only)
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-4")
    content_tokens = len(encoding.encode(text))

    # calculate_tokens should add overhead for message formatting
    result = core.calculate_tokens(text, "gpt-4o")

    # Should be more than just content tokens (includes ~3 token overhead)
    assert result > content_tokens
    assert result == content_tokens + 3  # OpenAI's per-message overhead
