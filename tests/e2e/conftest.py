"""Shared fixtures for e2e tests."""

import pytest

from multi_llm_chat.core import AgenticLoopResult


@pytest.fixture
def fake_stream_factory():
    """Return a factory that creates fake execute_with_tools_stream functions."""

    def _make(message: str):
        async def _fake_stream(provider, input_history, system_prompt, **kwargs):
            yield {"type": "text", "content": message}
            yield AgenticLoopResult(
                chunks=[{"type": "text", "content": message}],
                history_delta=[
                    {"role": "gemini", "content": [{"type": "text", "content": message}]}
                ],
                final_text=message,
                iterations_used=1,
                timed_out=False,
            )

        return _fake_stream

    return _make


@pytest.fixture
def dummy_provider():
    class DummyProvider:
        """Minimal provider placeholder for tests."""

        pass

    return DummyProvider()
