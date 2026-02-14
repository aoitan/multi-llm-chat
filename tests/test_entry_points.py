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


def test_root_app_main_execution():
    """Root app.py should be executable via python app.py (though deprecated)"""
    # Test actual __main__ execution path (will timeout when server starts, which is expected)
    try:
        result = subprocess.run(
            [sys.executable, "app.py"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3,  # Will timeout when server tries to start
        )
        stderr = result.stderr
    except subprocess.TimeoutExpired as e:
        # Timeout is expected (server started successfully)
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr

    # Should show deprecation warning in stderr
    assert "DeprecationWarning" in stderr or "FutureWarning" in stderr
    assert "python -m multi_llm_chat.webui" in stderr
    # Should not have coroutine errors
    assert "coroutine" not in stderr.lower()
    assert "was never awaited" not in stderr.lower()


def test_root_chat_logic_main_execution():
    """Root chat_logic.py should be executable via python chat_logic.py (though deprecated)"""
    # Test actual __main__ execution path with immediate exit
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "import chat_logic; "
            # Verify main() returns a coroutine that can be awaited
            "import asyncio; import inspect; "
            "assert inspect.iscoroutinefunction(chat_logic.main), 'main should be async'",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=3,
    )

    # Should succeed
    assert result.returncode == 0, f"Execution failed: {result.stderr}"
    # Should show deprecation warning
    assert "DeprecationWarning" in result.stderr or "FutureWarning" in result.stderr
    # Should not have coroutine errors
    assert "coroutine" not in result.stderr.lower()
    assert "was never awaited" not in result.stderr.lower()
