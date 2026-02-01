import pytest

pytest_plugins = ["tests.conftest_llm"]


def pytest_configure(config):
    """Initialize runtime before test collection (pytest plugin hook)."""
    from multi_llm_chat.runtime import init_runtime, is_initialized

    if not is_initialized():
        init_runtime()


@pytest.fixture(autouse=True)
def ensure_config_initialized():
    """Ensure configuration is initialized before each test."""
    from multi_llm_chat.config import (
        is_config_initialized,
        load_config_from_env,
        set_config,
    )

    # If config was reset by a previous test, reinitialize it
    if not is_config_initialized():
        config = load_config_from_env()
        set_config(config)

    yield

    # Clean up is optional - tests that need isolation will handle it themselves


async def collect_async_generator(async_gen):
    """Helper to collect async generator results into a list"""
    results = []
    async for item in async_gen:
        results.append(item)
    return results


# ========================================
# Shared test fixtures
# ========================================
# (none currently; add here when needed)
