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

    # Should succeed without errors
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "Traceback" not in result.stderr, f"Unexpected error: {result.stderr}"

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

    # Should succeed without errors
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "Traceback" not in result.stderr, f"Unexpected error: {result.stderr}"

    # Should contain deprecation message in stderr
    assert "DeprecationWarning" in result.stderr
    assert "python -m multi_llm_chat.cli" in result.stderr
    assert "v2.0.0" in result.stderr


def test_package_app_module_deprecation_warning():
    """Package multi_llm_chat.app module should emit deprecation warning when imported"""
    # Use subprocess to ensure fresh import (avoid sys.modules cache issues)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); "
            "import warnings; warnings.simplefilter('always'); "
            "import multi_llm_chat.app",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should succeed without errors
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "Traceback" not in result.stderr, f"Unexpected error: {result.stderr}"

    # Should contain deprecation message in stderr
    assert "DeprecationWarning" in result.stderr
    assert "multi_llm_chat.app" in result.stderr
    assert "multi_llm_chat.webui" in result.stderr
    assert "python -m multi_llm_chat.webui" in result.stderr


def test_package_chat_logic_module_deprecation_warning():
    """Package multi_llm_chat.chat_logic module should emit deprecation warning when imported"""
    # Use subprocess to ensure fresh import (avoid sys.modules cache issues)
    # Note: chat_logic import may fail due to config requirements,
    # but deprecation warning should still appear
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); "
            "import warnings; warnings.simplefilter('always'); "
            "import multi_llm_chat.chat_logic",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Deprecation warning should appear in stderr (even if import fails later)
    assert "DeprecationWarning" in result.stderr
    assert "multi_llm_chat.chat_logic" in result.stderr
    assert "multi_llm_chat.chat_service" in result.stderr
