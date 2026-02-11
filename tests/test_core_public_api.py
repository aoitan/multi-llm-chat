"""Tests for core.py - Public API validation

This module tests the official public API exported by core.py (__all__).
Tests cover:
- Token and context management
- Agentic Loop functionality
- Provider classes and configuration
- History utilities

Legacy API tests are in test_core_facade.py and test_core_legacy_api.py.
"""


def test_core_exports_token_management_api():
    """core.py should export token and context management functions"""
    import multi_llm_chat.core as core

    # All token management functions should be exported
    assert hasattr(core, "get_token_info")
    assert hasattr(core, "calculate_tokens")
    assert hasattr(core, "get_max_context_length")
    assert hasattr(core, "get_pruning_info")
    assert hasattr(core, "prune_history_sliding_window")
    assert hasattr(core, "validate_context_length")
    assert hasattr(core, "validate_system_prompt_length")


def test_core_exports_agentic_loop_api():
    """core.py should export Agentic Loop functions and types"""
    import multi_llm_chat.core as core

    # Agentic Loop functions
    assert hasattr(core, "execute_with_tools")
    assert hasattr(core, "execute_with_tools_stream")
    assert hasattr(core, "execute_with_tools_sync")

    # Agentic Loop result type
    assert hasattr(core, "AgenticLoopResult")


def test_core_exports_provider_api():
    """core.py should export provider classes and utilities"""
    import multi_llm_chat.core as core

    # Provider classes
    assert hasattr(core, "GeminiProvider")
    assert hasattr(core, "ChatGPTProvider")

    # Provider factory
    assert hasattr(core, "create_provider")
    assert hasattr(core, "get_provider")

    # Provider utilities
    assert hasattr(core, "list_gemini_models")


def test_core_exports_history_utils():
    """core.py should export history utilities"""
    import multi_llm_chat.core as core

    assert hasattr(core, "LLM_ROLES")


def test_token_info_returns_correct_structure():
    """get_token_info() should return dict with required fields"""
    import multi_llm_chat.core as core

    result = core.get_token_info("test", "gemini-2.0-flash-exp")
    assert isinstance(result, dict)
    assert "token_count" in result
    assert "max_context_length" in result
    assert "is_estimated" in result
    assert isinstance(result["token_count"], int)
    assert isinstance(result["max_context_length"], int)
    assert isinstance(result["is_estimated"], bool)


def test_agentic_loop_functions_are_callable():
    """Agentic Loop functions should be callable"""
    import inspect

    import multi_llm_chat.core as core

    assert callable(core.execute_with_tools)
    assert callable(core.execute_with_tools_stream)
    assert callable(core.execute_with_tools_sync)

    # AgenticLoopResult should be a class
    assert inspect.isclass(core.AgenticLoopResult)


def test_provider_classes_are_instantiable():
    """Provider classes should be accessible and have expected interface"""
    import inspect

    import multi_llm_chat.core as core

    # Both providers should be classes
    assert inspect.isclass(core.GeminiProvider)
    assert inspect.isclass(core.ChatGPTProvider)

    # Factory functions should be callable
    assert callable(core.create_provider)
    assert callable(core.get_provider)


def test_core_all_contains_only_public_api():
    """core.__all__ should NOT contain legacy API functions"""
    import multi_llm_chat.core as core

    # Verify __all__ exists
    assert hasattr(core, "__all__")
    all_exports = core.__all__

    # Legacy API functions should NOT be in __all__
    legacy_functions = [
        "call_gemini_api",
        "call_chatgpt_api",
        "call_gemini_api_async",
        "call_chatgpt_api_async",
        "format_history_for_gemini",
        "format_history_for_chatgpt",
        "load_api_key",
        "prepare_request",
        "stream_text_events",
        "stream_text_events_async",
        "extract_text_from_chunk",
    ]
    for func in legacy_functions:
        assert func not in all_exports, f"{func} should not be in core.__all__"

    # Public API functions should be in __all__
    assert "get_token_info" in all_exports
    assert "execute_with_tools" in all_exports
    assert "GeminiProvider" in all_exports
    assert "create_provider" in all_exports


def test_legacy_api_still_importable_directly():
    """Legacy API should still be importable via core, but not in __all__"""
    import multi_llm_chat.core as core

    # Legacy functions should still be accessible (backward compatibility)
    assert hasattr(core, "call_gemini_api")
    assert hasattr(core, "call_chatgpt_api")
    assert hasattr(core, "format_history_for_gemini")

    # But they should NOT be in __all__
    assert "call_gemini_api" not in core.__all__
