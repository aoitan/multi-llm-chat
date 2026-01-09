import pytest
from multi_llm_chat.compression import prune_history_sliding_window, get_pruning_info

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
    max_tokens = 500

    pruned = prune_history_sliding_window(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
    )

    # Should keep only the most recent turns that fit
    assert len(pruned) < len(history)
    assert len(pruned) >= 2
    assert pruned[-1]["content"] == history[-1]["content"]

def test_pruning_by_turn_pairs():
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "gemini", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "gemini", "content": "A2"},
        {"role": "user", "content": "Q3"},
    ]

    max_tokens = 30

    pruned = prune_history_sliding_window(
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
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
        history, max_tokens, model_name="gemini-1.5-pro", system_prompt=None
    )

    assert "turns_to_remove" in info
    assert info["turns_to_remove"] > 0
    assert "original_length" in info
    assert "pruned_length" in info
