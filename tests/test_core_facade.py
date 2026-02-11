"""Tests for core.py - Legacy API backward compatibility validation

⚠️ LEGACY API TESTS - These functions are deprecated and will be removed in v2.0.0
This module tests that legacy APIs are still accessible via core.py for backward compatibility.
These tests ensure existing code doesn't break immediately.

For public API tests, see test_core_public_api.py
For legacy API behavior tests, see test_core_legacy_api.py
"""

import pytest


@pytest.mark.legacy
def test_core_module_exports_legacy_api_functions():
    """core.py should still allow importing legacy API functions (not in __all__)"""
    import multi_llm_chat.core as core

    # Legacy API functions should be accessible via direct import
    # (but not via `from core import *` since they're not in __all__)
    assert hasattr(core, "load_api_key")
    assert hasattr(core, "format_history_for_gemini")
    assert hasattr(core, "format_history_for_chatgpt")
    assert hasattr(core, "prepare_request")
    assert hasattr(core, "call_gemini_api")
    assert hasattr(core, "call_chatgpt_api")
    assert hasattr(core, "call_gemini_api_async")
    assert hasattr(core, "call_chatgpt_api_async")
    assert hasattr(core, "stream_text_events")
    assert hasattr(core, "stream_text_events_async")
    assert hasattr(core, "extract_text_from_chunk")


@pytest.mark.legacy
def test_legacy_functions_are_callable():
    """Legacy functions should still be callable for backward compatibility"""
    import multi_llm_chat.core as core

    # Just verify they're callable - actual behavior tested in test_core_legacy_api.py
    assert callable(core.call_gemini_api)
    assert callable(core.call_chatgpt_api)
    assert callable(core.format_history_for_gemini)
    assert callable(core.format_history_for_chatgpt)
