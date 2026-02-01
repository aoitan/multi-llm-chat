"""Tests for runtime initialization module."""

import logging
from unittest.mock import patch

import pytest

from multi_llm_chat.config import is_config_initialized, reset_config
from multi_llm_chat.runtime import init_runtime, is_initialized, reset_runtime


@pytest.fixture(autouse=True)
def reset_state():
    """Reset runtime and config state before each test."""
    reset_runtime()
    reset_config()
    yield
    reset_runtime()
    reset_config()


def test_init_runtime_loads_dotenv():
    """Test that init_runtime() calls load_dotenv() and initializes config."""
    with patch("multi_llm_chat.runtime.load_dotenv") as mock_load:
        init_runtime()
        mock_load.assert_called_once()
        assert is_initialized()
        assert is_config_initialized()  # Config should also be initialized


def test_init_runtime_idempotent():
    """Test that calling init_runtime() multiple times is safe."""
    with patch("multi_llm_chat.runtime.load_dotenv") as mock_load:
        init_runtime()
        init_runtime()
        init_runtime()
        # load_dotenv should only be called once
        mock_load.assert_called_once()
        assert is_initialized()


def test_init_runtime_with_log_level():
    """Test that init_runtime() configures logging when log_level is provided."""
    with (
        patch("multi_llm_chat.runtime.load_dotenv"),
        patch("multi_llm_chat.runtime.logging.basicConfig") as mock_config,
    ):
        init_runtime(log_level="DEBUG")
        mock_config.assert_called_once_with(level=logging.DEBUG)


def test_init_runtime_with_invalid_log_level():
    """Test that init_runtime() raises ValueError for invalid log level."""
    with patch("multi_llm_chat.runtime.load_dotenv"):
        with pytest.raises(ValueError, match="Invalid log level"):
            init_runtime(log_level="INVALID")


def test_init_runtime_without_log_level():
    """Test that init_runtime() does not configure logging when log_level is None."""
    with (
        patch("multi_llm_chat.runtime.load_dotenv"),
        patch("multi_llm_chat.runtime.logging.basicConfig") as mock_config,
    ):
        init_runtime(log_level=None)
        mock_config.assert_not_called()


def test_is_initialized_before_init():
    """Test that is_initialized() returns False before init_runtime()."""
    assert not is_initialized()


def test_is_initialized_after_init():
    """Test that is_initialized() returns True after init_runtime()."""
    with patch("multi_llm_chat.runtime.load_dotenv"):
        init_runtime()
        assert is_initialized()


def test_reset_runtime():
    """Test that reset_runtime() resets initialization state."""
    with patch("multi_llm_chat.runtime.load_dotenv"):
        init_runtime()
        assert is_initialized()
        reset_runtime()
        assert not is_initialized()


def test_init_runtime_invalid_log_level_leaves_uninitialized():
    """Test that invalid log level raises error and leaves runtime uninitialized."""
    with patch("multi_llm_chat.runtime.load_dotenv"):
        with pytest.raises(ValueError, match="Invalid log level"):
            init_runtime(log_level="INVALID")
        # Runtime should remain uninitialized after error
        assert not is_initialized()


def test_init_runtime_invalid_then_valid():
    """Test that runtime can be initialized after a failed attempt."""
    with patch("multi_llm_chat.runtime.load_dotenv"):
        # First attempt fails
        with pytest.raises(ValueError, match="Invalid log level"):
            init_runtime(log_level="INVALID")
        assert not is_initialized()

        # Second attempt succeeds
        init_runtime(log_level="DEBUG")
        assert is_initialized()


def test_init_runtime_thread_safety():
    """Test that init_runtime() is thread-safe with concurrent calls."""
    import threading

    call_count = [0]
    original_load_dotenv = __import__("dotenv").load_dotenv

    def counting_load_dotenv(*args, **kwargs):
        call_count[0] += 1
        return original_load_dotenv(*args, **kwargs)

    with patch("multi_llm_chat.runtime.load_dotenv", side_effect=counting_load_dotenv):
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=init_runtime)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # load_dotenv should only be called once despite 10 concurrent calls
        assert call_count[0] == 1
        assert is_initialized()
