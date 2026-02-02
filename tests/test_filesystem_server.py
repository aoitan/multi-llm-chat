"""Tests for filesystem MCP server configuration."""

import logging
from pathlib import Path
from unittest.mock import patch

from multi_llm_chat.mcp.filesystem_server import (
    create_filesystem_server_config,
    is_dangerous_path,
)


class TestIsDangerousPath:
    """Tests for dangerous path detection."""

    def test_is_dangerous_path_root(self):
        """Test that root directory is detected as dangerous."""
        assert is_dangerous_path("/") is True

    def test_is_dangerous_path_home(self):
        """Test that home directory is detected as dangerous."""
        home = str(Path.home())
        assert is_dangerous_path(home) is True

    def test_is_dangerous_path_safe(self):
        """Test that safe paths are not flagged as dangerous."""
        assert is_dangerous_path("/usr/local/myproject") is False
        assert is_dangerous_path("/tmp/test") is False

    def test_is_dangerous_path_home_subdirectory(self):
        """Test that subdirectories of home are safe."""
        home_subdir = str(Path.home() / "projects" / "myapp")
        assert is_dangerous_path(home_subdir) is False


class TestCreateFilesystemServerConfig:
    """Tests for filesystem server configuration factory."""

    def test_create_config_explicit_root(self):
        """Test creating config with explicit root directory."""
        config = create_filesystem_server_config("/custom/path")
        assert config.name == "filesystem"
        assert config.server_args == ["mcp-server-filesystem", "/custom/path"]

    def test_create_config_default_cwd(self):
        """Test creating config defaults to current working directory."""
        with patch("os.getcwd", return_value="/mock/cwd"):
            config = create_filesystem_server_config()
            assert config.server_args == ["mcp-server-filesystem", "/mock/cwd"]

    def test_create_config_warns_dangerous_path(self, caplog):
        """Test that dangerous paths trigger warning log."""
        with caplog.at_level(logging.WARNING):
            create_filesystem_server_config("/")
            assert any("dangerous" in record.message.lower() for record in caplog.records)

    def test_create_config_command(self):
        """Test that config uses uvx command."""
        config = create_filesystem_server_config("/test/path")
        assert config.server_command == "uvx"
        assert "mcp-server-filesystem" in config.server_args

    def test_create_config_timeout(self):
        """Test that config uses default MCP timeout."""
        config = create_filesystem_server_config("/test/path")
        assert config.timeout == 120
