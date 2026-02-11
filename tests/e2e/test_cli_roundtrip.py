import pytest

from multi_llm_chat.chat_service import ChatService


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_chatservice_roundtrip_with_mocked_provider(
    monkeypatch, fake_stream_factory, dummy_provider
):
    """ChatServiceがモックプロバイダ経由で履歴を更新する簡易E2E。"""

    fake_stream = fake_stream_factory("Hello from mock!")

    monkeypatch.setattr("multi_llm_chat.core.execute_with_tools_stream", fake_stream)
    monkeypatch.setattr("multi_llm_chat.chat_service.create_provider", lambda name: dummy_provider)

    service = ChatService()

    # Collect final state from the stream
    async for _display_history, _logic_history, _chunk in service.process_message("@gemini hi"):
        pass

    assert service.logic_history[-1]["role"] == "gemini"
    assert service.logic_history[-1]["content"][0]["content"] == "Hello from mock!"
    assert service.display_history[-1][1].startswith("**Gemini:**")
    assert "Hello from mock!" in service.display_history[-1][1]
