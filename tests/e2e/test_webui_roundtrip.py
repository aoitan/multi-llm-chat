import pytest

from multi_llm_chat.webui.handlers import validate_and_respond


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_webui_validate_and_respond_roundtrip(
    monkeypatch, fake_stream_factory, dummy_provider
):
    """WebUIの応答パイプラインがモックプロバイダで履歴を更新するかを確認する。"""

    fake_stream = fake_stream_factory("Hello WebUI!")

    monkeypatch.setattr("multi_llm_chat.chat_service.execute_with_tools_stream", fake_stream)
    monkeypatch.setattr("multi_llm_chat.chat_service.create_provider", lambda name: dummy_provider)

    display_history = []
    logic_history = []
    system_prompt = ""
    chat_service = None

    async for (
        _display_history,
        _display_state,
        _logic_history,
        _chat_service,
    ) in validate_and_respond(
        "@gemini hi",
        display_history,
        logic_history,
        system_prompt,
        "user1",
        chat_service,
    ):
        pass

    assert logic_history[-1]["role"] == "gemini"
    assert logic_history[-1]["content"][0]["content"] == "Hello WebUI!"
    assert display_history[-1][1].startswith("**Gemini:**")
    assert "Hello WebUI!" in display_history[-1][1]
