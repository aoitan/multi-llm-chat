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
