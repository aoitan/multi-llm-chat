"""Tests for root entry point deprecation warnings (Issue #116)

This module tests that the root app.py and chat_logic.py emit
deprecation warnings when executed directly.
"""

import subprocess
import sys


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
        cwd="/Users/aoitan/workspace/multi-llm-chat/copilot",
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
        cwd="/Users/aoitan/workspace/multi-llm-chat/copilot",
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should contain deprecation message in stderr
    assert "DeprecationWarning" in result.stderr
    assert "python -m multi_llm_chat.cli" in result.stderr
    assert "v2.0.0" in result.stderr
