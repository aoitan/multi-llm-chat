from multi_llm_chat.token_utils import estimate_tokens, get_buffer_factor, get_max_context_length


def test_estimate_tokens():
    # ASCII text: ~4 chars = 1 token
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2

    # Japanese text: ~1.5 chars = 1 token
    assert estimate_tokens("あいう") == 2
    assert estimate_tokens("あいうえお") == 3

    # Mixed
    # "Hello" (5 chars) -> 5/4 = 1.25
    # "こんにちは" (5 chars) -> 5/1.5 = 3.33
    # Total = 4.58 -> 4 (int conversion)
    assert estimate_tokens("Helloこんにちは") == 4


def test_estimate_tokens_with_structured_content():
    content = [
        {"type": "text", "content": "abcd"},
        {"type": "text", "content": "efgh"},
    ]

    assert estimate_tokens(content) == 2


def test_estimate_tokens_with_tool_calls_under_buffer_factor():
    """ツール呼び出しを含むcontentのトークン推定が妥当な範囲内であること

    Note: This test verifies that token estimation for tool calls produces
    reasonable values. Actual API token counts may differ due to schema
    metadata included by Gemini. BUFFER_FACTOR compensates for this.
    """
    from multi_llm_chat.history_utils import content_to_text

    # Simulate large tool call with nested JSON
    large_tool_call = {
        "type": "tool_call",
        "content": {
            "name": "search_documents",
            "arguments": {
                "query": "artificial intelligence " * 50,  # ~100 tokens worth
                "filters": {"date_range": "2020-2025", "category": ["research", "engineering"]},
            },
        },
    }
    content = [large_tool_call]

    # Estimate tokens using content_to_text
    text_repr = content_to_text(content, include_tool_data=True)
    estimated_tokens = estimate_tokens(text_repr)

    # Verify estimation produces reasonable values
    assert estimated_tokens > 0, "Tool call token count should be positive"
    assert estimated_tokens < 10000, "Tool call token count should be reasonable (sanity check)"

    # Verify the text representation includes tool data
    assert "search_documents" in text_repr
    assert "artificial intelligence" in text_repr

    # TODO: Add integration test with actual Gemini API token count
    # to verify BUFFER_FACTOR adequacy in practice


def test_get_max_context_length(monkeypatch):
    # Clear environment variables to test defaults
    monkeypatch.delenv("GEMINI_MAX_CONTEXT_LENGTH", raising=False)
    monkeypatch.delenv("CHATGPT_MAX_CONTEXT_LENGTH", raising=False)
    monkeypatch.delenv("DEFAULT_MAX_CONTEXT_LENGTH", raising=False)

    # Default for gemini
    assert get_max_context_length("gemini-pro") == 32760

    # Environment variable override
    monkeypatch.setenv("GEMINI_MAX_CONTEXT_LENGTH", "1000")
    assert get_max_context_length("gemini-pro") == 1000

    # Default for unknown
    monkeypatch.delenv("GEMINI_MAX_CONTEXT_LENGTH", raising=False)
    assert get_max_context_length("unknown-model") == 4096


def test_get_buffer_factor(monkeypatch):
    # Clear any existing environment variable first
    monkeypatch.delenv("TOKEN_ESTIMATION_BUFFER_FACTOR", raising=False)

    # Default without tools (1.2 for standard conversation)
    assert get_buffer_factor(has_tools=False) == 1.2

    # Default with tools (1.5 to account for FunctionDeclaration overhead)
    assert get_buffer_factor(has_tools=True) == 1.5

    # Environment variable overrides everything
    monkeypatch.setenv("TOKEN_ESTIMATION_BUFFER_FACTOR", "2.0")
    assert get_buffer_factor(has_tools=False) == 2.0
    assert get_buffer_factor(has_tools=True) == 2.0


def test_get_buffer_factor_invalid_env_value(monkeypatch):
    """Invalid environment variable should fall back to auto-detected default"""
    monkeypatch.setenv("TOKEN_ESTIMATION_BUFFER_FACTOR", "invalid")

    # Should fall back to auto-detected values
    assert get_buffer_factor(has_tools=False) == 1.2
    assert get_buffer_factor(has_tools=True) == 1.5
