import pytest

pytest_plugins = ["tests.conftest_llm"]


async def collect_async_generator(async_gen):
    """Helper to collect async generator results into a list"""
    results = []
    async for item in async_gen:
        results.append(item)
    return results


# ========================================
# Shared test fixtures for core/history/validation tests
# ========================================


@pytest.fixture
def sample_history_basic():
    """Basic conversation history with user and assistant turns"""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


@pytest.fixture
def sample_history_gemini():
    """Conversation history with Gemini responses"""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "gemini", "content": "Hi there"},
    ]


@pytest.fixture
def sample_history_chatgpt():
    """Conversation history with ChatGPT responses"""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "chatgpt", "content": "Hi there"},
    ]


@pytest.fixture
def sample_history_mixed():
    """Conversation history with both Gemini and ChatGPT responses"""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "chatgpt", "content": "Hi from ChatGPT"},
        {"role": "user", "content": "Another question"},
        {"role": "gemini", "content": "Answer from Gemini"},
    ]
