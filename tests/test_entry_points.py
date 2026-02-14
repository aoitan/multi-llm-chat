"""Tests for root entry point deprecation warnings (Issue #116)

This module tests that the root app.py and chat_logic.py emit
deprecation warnings when executed directly.
"""

import subprocess
import sys
from pathlib import Path

# Dynamically get repository root (parent of tests directory)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_app_deprecation_warning():
    """Root app.py should emit deprecation warning when executed"""
    # Import the module to trigger the warning without starting the server
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "import warnings; warnings.simplefilter('always'); "
            "import app",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should contain deprecation message in stderr
    assert "DeprecationWarning" in result.stderr
    assert "python -m multi_llm_chat.webui" in result.stderr
    assert "v2.0.0" in result.stderr


def test_root_chat_logic_deprecation_warning():
    """Root chat_logic.py should emit deprecation warning when executed"""
    # Import the module to trigger the warning without starting the REPL
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "import warnings; warnings.simplefilter('always'); "
            "import chat_logic",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should contain deprecation message in stderr
    assert "DeprecationWarning" in result.stderr
    assert "python -m multi_llm_chat.cli" in result.stderr
    assert "v2.0.0" in result.stderr


def test_package_app_module_deprecation_warning():
    """Package multi_llm_chat.app module should emit deprecation warning when imported"""
    import sys
    import warnings

    # Skip test if module is already imported (avoid cache issues in full test suite)
    if "multi_llm_chat.app" in sys.modules:
        import pytest

        pytest.skip("Module already imported, cannot test fresh import warning")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Import the package module (not the root script)
        import multi_llm_chat.app  # noqa: F401

        # Should emit at least one DeprecationWarning for multi_llm_chat.app
        app_warnings = [warning for warning in w if "multi_llm_chat.app" in str(warning.message)]
        assert len(app_warnings) >= 1
        assert issubclass(app_warnings[0].category, DeprecationWarning)
        assert "multi_llm_chat.webui" in str(app_warnings[0].message)
        assert "python -m multi_llm_chat.webui" in str(app_warnings[0].message)


def test_package_chat_logic_module_deprecation_warning():
    """Package multi_llm_chat.chat_logic module should emit deprecation warning when imported"""
    import sys
    import warnings

    # Skip test if module is already imported (avoid cache issues in full test suite)
    if "multi_llm_chat.chat_logic" in sys.modules:
        import pytest

        pytest.skip("Module already imported, cannot test fresh import warning")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Import the package module (not the root script)
        import multi_llm_chat.chat_logic  # noqa: F401

        # Should emit at least one DeprecationWarning for multi_llm_chat.chat_logic
        chat_logic_warnings = [
            warning for warning in w if "multi_llm_chat.chat_logic" in str(warning.message)
        ]
        assert len(chat_logic_warnings) >= 1
        assert issubclass(chat_logic_warnings[0].category, DeprecationWarning)
        assert "multi_llm_chat.chat_service" in str(chat_logic_warnings[0].message)
