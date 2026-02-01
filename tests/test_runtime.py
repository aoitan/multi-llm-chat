"""Tests for runtime initialization module."""

import logging
from unittest.mock import patch

import pytest

from multi_llm_chat.runtime import init_runtime, is_initialized, reset_runtime


@pytest.fixture(autouse=True)
def reset_state():
    """Reset runtime state before each test."""
    reset_runtime()
    yield
    reset_runtime()


def test_init_runtime_loads_dotenv():
    """Test that init_runtime() calls load_dotenv()."""
    with patch("multi_llm_chat.runtime.load_dotenv") as mock_load:
        init_runtime()
        mock_load.assert_called_once()
        assert is_initialized()


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
