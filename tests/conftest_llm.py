"""Shared fixtures and mocks for LLM provider tests

This module centralizes common test fixtures for OpenAI and Gemini providers,
reducing duplication across test_llm_provider_*.py files.
Created as part of Issue #101 provider refactoring.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_openai_api_key(monkeypatch):
    """Patch OpenAI API key to avoid environment dependency"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def mock_google_api_key(monkeypatch):
    """Patch Google API key to avoid environment dependency"""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with common configuration"""
    mock_client = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([MagicMock()]))
    mock_client.chat.completions.create.return_value = mock_stream
    return mock_client


@pytest.fixture
def mock_gemini_model():
    """Create a mock Gemini GenerativeModel with common configuration"""
    mock_model = MagicMock()
    mock_model.generate_content.return_value = iter([])
    return mock_model


@pytest.fixture
def basic_history():
    """Standard conversation history for testing"""
    return [{"role": "user", "content": "Hello"}]
