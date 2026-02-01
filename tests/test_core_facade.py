"""Tests for core.py - Facade pattern validation

This module tests that the core.py module correctly acts as a facade,
re-exporting functionality from core_modules without breaking backward compatibility.
These tests verify that the public API surface remains stable.
"""


def test_core_module_exports_expected_symbols():
    """core.py should re-export all expected public symbols"""
    import multi_llm_chat.core as core

    # Token & Context utilities
    assert hasattr(core, "get_token_info")
    assert hasattr(core, "calculate_tokens")
    assert hasattr(core, "get_max_context_length")

    # Legacy API functions
    assert hasattr(core, "load_api_key")
    assert hasattr(core, "format_history_for_gemini")
    assert hasattr(core, "format_history_for_chatgpt")
    assert hasattr(core, "prepare_request")
    assert hasattr(core, "call_gemini_api")
    assert hasattr(core, "call_chatgpt_api")
    assert hasattr(core, "stream_text_events")
    assert hasattr(core, "extract_text_from_chunk")

    # Agentic Loop
    assert hasattr(core, "execute_with_tools")
    assert hasattr(core, "execute_with_tools_stream")
    assert hasattr(core, "execute_with_tools_sync")
    assert hasattr(core, "AgenticLoopResult")


def test_core_module_backward_compatibility_types():
    """core.py should maintain backward compatible type signatures"""
    import inspect

    import multi_llm_chat.core as core

    # Verify get_token_info returns dict
    result = core.get_token_info("test", "gemini-2.0-flash-exp")
    assert isinstance(result, dict)
    assert "token_count" in result
    assert "max_context_length" in result
    assert "is_estimated" in result

    # Verify execute_with_tools (agentic loop) is callable
    assert callable(core.execute_with_tools)
    assert callable(core.execute_with_tools_stream)
    assert callable(core.execute_with_tools_sync)

    # Verify AgenticLoopResult is a class
    assert inspect.isclass(core.AgenticLoopResult)
