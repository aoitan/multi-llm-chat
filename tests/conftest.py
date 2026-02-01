pytest_plugins = ["tests.conftest_llm"]


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
