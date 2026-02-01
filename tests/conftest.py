pytest_plugins = ["tests.conftest_llm"]


def pytest_configure(config):
    """Initialize runtime before test collection (pytest plugin hook)."""
    from multi_llm_chat.runtime import init_runtime, is_initialized

    if not is_initialized():
        init_runtime()


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
