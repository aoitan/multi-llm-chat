from multi_llm_chat.compression import get_pruning_info, prune_history_sliding_window


def mock_calculate_tokens(text: str, model_name: str) -> int:
    """Mock token calculation: 1 char = 1 token for simplicity"""
    return len(text)


def test_sliding_window_pruning_basic():
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
    # Mock calculation: 1 char = 1 token
    # Length of one message is approx 440 chars ("This is..." * 10)
    # One turn (user + assistant) is approx 880 tokens

    # Set limit to fit only the latest turn
    max_tokens = 1000

    pruned = prune_history_sliding_window(
        history,
        max_tokens,
        model_name="gemini-1.5-pro",
        system_prompt=None,
        token_calculator=mock_calculate_tokens,
    )

    # Should keep only the most recent turn (1 turn = 2 messages)
    assert len(pruned) == 2
    assert pruned[-1]["content"] == history[-1]["content"]


def test_pruning_by_turn_pairs():
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "gemini", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "gemini", "content": "A2"},
        {"role": "user", "content": "Q3"},
    ]

    # Each "Qx" or "Ax" is 2 tokens
    # Total: 10 tokens
    # Limit: 5 tokens (should keep Q3 + maybe A2/Q2 pair if fits?)
    # Wait, simple mock is len(text)
    # Q1=2, A1=2, Q2=2, A2=2, Q3=2

    max_tokens = 5

    pruned = prune_history_sliding_window(
        history,
        max_tokens,
        model_name="gemini-1.5-pro",
        system_prompt=None,
        token_calculator=mock_calculate_tokens,
    )

    if len(pruned) < len(history):
        for i, entry in enumerate(pruned):
            if entry["role"] in ["gemini", "chatgpt"]:
                assert i > 0
                assert pruned[i - 1]["role"] == "user"


def test_get_pruning_info():
    history = [
        {"role": "user", "content": "Message " + "x" * 100},
        {"role": "gemini", "content": "Response " + "y" * 100},
        {"role": "user", "content": "Message " + "z" * 100},
        {"role": "gemini", "content": "Response " + "w" * 100},
    ]

    max_tokens = 50

    info = get_pruning_info(
        history,
        max_tokens,
        model_name="gemini-1.5-pro",
        system_prompt=None,
        token_calculator=mock_calculate_tokens,
    )

    assert "turns_to_remove" in info
    assert info["turns_to_remove"] > 0
    assert "original_length" in info
    assert "pruned_length" in info
