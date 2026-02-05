import pytest

from multi_llm_chat.chat_logic import ChatService
from multi_llm_chat.core import AgenticLoopResult


@pytest.mark.anyio
@pytest.mark.e2e
async def test_chatservice_roundtrip_with_mocked_provider(monkeypatch):
    """ChatServiceがモックプロバイダ経由で履歴を更新する簡易E2E。"""

    async def fake_stream(provider, input_history, system_prompt, **kwargs):
        yield {"type": "text", "content": "Hello from mock!"}
        yield AgenticLoopResult(
            chunks=[{"type": "text", "content": "Hello from mock!"}],
            history_delta=[
                {"role": "gemini", "content": [{"type": "text", "content": "Hello from mock!"}]}
            ],
            final_text="Hello from mock!",
            iterations_used=1,
            timed_out=False,
        )

    class DummyProvider:
        """最小限のダミープロバイダ。"""

        pass

    monkeypatch.setattr(
        "multi_llm_chat.core.execute_with_tools_stream",
        fake_stream,
    )
    monkeypatch.setattr(
        "multi_llm_chat.chat_logic.create_provider",
        lambda name: DummyProvider(),
    )

    service = ChatService()

    # Collect final state from the stream
    async for display_history, logic_history, _chunk in service.process_message("@gemini hi"):
        pass

    assert logic_history[-1]["role"] == "gemini"
    assert logic_history[-1]["content"][0]["content"] == "Hello from mock!"
    assert display_history[-1][1].startswith("**Gemini:**")
    assert "Hello from mock!" in display_history[-1][1]
