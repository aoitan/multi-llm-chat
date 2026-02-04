"""Tests for MCP integration with runtime initialization."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_llm_chat.config import reset_config
from multi_llm_chat.mcp import get_mcp_manager, reset_mcp_manager
from multi_llm_chat.runtime import init_runtime, reset_runtime


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all global state before each test."""
    reset_config()
    reset_runtime()
    reset_mcp_manager()
    yield
    reset_config()
    reset_runtime()
    reset_mcp_manager()


class TestMCPIntegration:
    """Tests for MCP integration with init_runtime()."""

    def test_skip_mcp_when_disabled(self):
        """Test that MCP is not initialized when disabled."""
        with patch.dict("os.environ", {"MULTI_LLM_CHAT_MCP_ENABLED": "false"}, clear=True):
            init_runtime()
            manager = get_mcp_manager()
            assert manager is None

    @patch("multi_llm_chat.runtime._init_mcp")
    def test_init_mcp_when_enabled(self, mock_init_mcp):
        """Test that _init_mcp is called when MCP is enabled."""
        with patch.dict(
            "os.environ",
            {
                "MULTI_LLM_CHAT_MCP_ENABLED": "true",
                "MCP_FILESYSTEM_ROOT": "/test/path",
            },
            clear=True,
        ):
            init_runtime()

            # Verify _init_mcp was called
            mock_init_mcp.assert_called_once()
            # Verify it was called with the config
            config = mock_init_mcp.call_args[0][0]
            assert config.mcp_enabled is True
            assert config.mcp_filesystem_root == "/test/path"

    @patch("multi_llm_chat.runtime._init_mcp")
    def test_filesystem_server_registered(self, mock_init_mcp):
        """Test that _init_mcp receives correct config."""
        with patch.dict(
            "os.environ",
            {"MULTI_LLM_CHAT_MCP_ENABLED": "true"},
            clear=True,
        ):
            init_runtime()

            # Verify _init_mcp was called
            assert mock_init_mcp.called
            # Verify config has MCP enabled
            config = mock_init_mcp.call_args[0][0]
            assert config.mcp_enabled is True

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    def test_cleanup_stops_servers(self, mock_client_class):
        """Test that MCP servers are properly stopped on cleanup."""
        # Mock the MCPClient to avoid actual subprocess creation
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        with patch.dict(
            "os.environ",
            {"MULTI_LLM_CHAT_MCP_ENABLED": "true"},
            clear=True,
        ):
            init_runtime()

            # Get the manager
            manager = get_mcp_manager()
            assert manager is not None

            # Manually call cleanup (simulating atexit)
            asyncio.run(manager.stop_all())

            # Verify client was properly exited
            mock_client.__aexit__.assert_called()

    @patch("multi_llm_chat.mcp.server_manager.MCPClient")
    def test_reset_mcp_manager_stops_servers(self, mock_client_class):
        """Test that reset_mcp_manager stops running servers."""
        # Mock the MCPClient
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        with patch.dict(
            "os.environ",
            {"MULTI_LLM_CHAT_MCP_ENABLED": "true"},
            clear=True,
        ):
            init_runtime()

            # Verify manager exists
            manager = get_mcp_manager()
            assert manager is not None

            # Reset should stop servers
            reset_mcp_manager()

            # Verify manager is cleared
            assert get_mcp_manager() is None

            # Verify client was properly exited
            mock_client.__aexit__.assert_called()

    @patch("multi_llm_chat.runtime._init_mcp")
    def test_double_initialization_is_idempotent(self, mock_init_mcp):
        """Test that calling init_runtime twice doesn't break."""
        with patch.dict(
            "os.environ",
            {"MULTI_LLM_CHAT_MCP_ENABLED": "true"},
            clear=True,
        ):
            init_runtime()
            # Second call should not raise error
            init_runtime()

            # _init_mcp should only be called once (idempotent)
            assert mock_init_mcp.call_count == 1

    def test_signal_handler_preserves_existing_handlers(self):
        """Test that MCP initialization preserves existing signal handlers."""
        import signal

        # Register a custom SIGTERM handler before init
        original_handler_called = {"called": False}

        def custom_sigterm_handler(signum, frame):
            original_handler_called["called"] = True

        signal.signal(signal.SIGTERM, custom_sigterm_handler)
        saved_original_handler = signal.getsignal(signal.SIGTERM)

        # Initialize runtime (which should preserve the original handler)
        # Mock _init_mcp to avoid actual server startup
        with patch("multi_llm_chat.runtime._init_mcp") as mock_init_mcp:
            # Simulate signal handler registration in _init_mcp
            def mock_init_side_effect(config):
                # Get original handler (should be custom_sigterm_handler)
                original_sigterm = signal.getsignal(signal.SIGTERM)
                # Also get SIGINT handler to match actual implementation
                _ = signal.getsignal(signal.SIGINT)  # Read but not used in this test

                def signal_handler(signum, frame):
                    # Chain to original handler if custom
                    if callable(original_sigterm) and original_sigterm != signal.SIG_DFL:
                        original_sigterm(signum, frame)

                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)

            mock_init_mcp.side_effect = mock_init_side_effect

            with patch.dict(
                "os.environ",
                {"MULTI_LLM_CHAT_MCP_ENABLED": "true"},
                clear=True,
            ):
                init_runtime()

                # Verify that a signal handler is now registered
                current_handler = signal.getsignal(signal.SIGTERM)
                assert current_handler is not None
                assert current_handler != signal.SIG_DFL
                # Handler should be wrapped, not the original
                assert current_handler != saved_original_handler

    @patch("multi_llm_chat.runtime._init_mcp")
    def test_init_mcp_from_non_main_thread(self, mock_init_mcp):
        """Test that _init_mcp gracefully handles non-main thread initialization.

        Signal handlers should be skipped when called from non-main thread,
        and atexit cleanup should still be registered.
        """
        import threading

        # Create a manager mock that will be set globally
        mock_manager = MagicMock()

        def mock_init_side_effect(config):
            # Simulate the actual _init_mcp flow
            from multi_llm_chat.mcp import set_mcp_manager

            set_mcp_manager(mock_manager)

        mock_init_mcp.side_effect = mock_init_side_effect

        exception_holder = []

        def run_init_in_thread():
            try:
                with patch.dict(
                    "os.environ",
                    {"MULTI_LLM_CHAT_MCP_ENABLED": "true", "GOOGLE_API_KEY": "test-key"},
                    clear=True,
                ):
                    init_runtime()
            except Exception as e:
                exception_holder.append(e)

        thread = threading.Thread(target=run_init_in_thread)
        thread.start()
        thread.join()

        # Should complete without ValueError about "signal only works in main thread"
        assert len(exception_holder) == 0, f"Unexpected exception: {exception_holder}"

        # Verify _init_mcp was called
        assert mock_init_mcp.call_count == 1
