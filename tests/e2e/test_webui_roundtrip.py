import pytest

from multi_llm_chat.core import AgenticLoopResult
from multi_llm_chat.webui.handlers import validate_and_respond


@pytest.mark.anyio
@pytest.mark.e2e
async def test_webui_validate_and_respond_roundtrip(monkeypatch):
    """WebUIの応答パイプラインがモックプロバイダで履歴を更新するかを確認する。"""

    async def fake_stream(provider, input_history, system_prompt, **kwargs):
        yield {"type": "text", "content": "Hello WebUI!"}
        yield AgenticLoopResult(
            chunks=[{"type": "text", "content": "Hello WebUI!"}],
            history_delta=[
                {"role": "gemini", "content": [{"type": "text", "content": "Hello WebUI!"}]}
            ],
            final_text="Hello WebUI!",
            iterations_used=1,
            timed_out=False,
        )

    class DummyProvider:
        pass

    monkeypatch.setattr(
        "multi_llm_chat.core.execute_with_tools_stream",
        fake_stream,
    )
    monkeypatch.setattr(
        "multi_llm_chat.chat_logic.create_provider",
        lambda name: DummyProvider(),
    )

    display_history = []
    logic_history = []
    system_prompt = ""
    chat_service = None

    async for display_history, _display_state, logic_history, chat_service in validate_and_respond(
        "@gemini hi", display_history, logic_history, system_prompt, "user1", chat_service
    ):
        pass

    assert logic_history[-1]["role"] == "gemini"
    assert logic_history[-1]["content"][0]["content"] == "Hello WebUI!"
    assert display_history[-1][1].startswith("**Gemini:**")
    assert "Hello WebUI!" in display_history[-1][1]
