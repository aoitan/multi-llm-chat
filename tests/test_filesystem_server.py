"""Tests for filesystem MCP server configuration."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        assert is_dangerous_path("/opt/myapp") is False

    def test_is_dangerous_path_home_subdirectory(self):
        """Test that subdirectories of home are safe."""
        home_subdir = str(Path.home() / "projects" / "myapp")
        assert is_dangerous_path(home_subdir) is False

    def test_is_dangerous_path_system_subdirectory_safe(self):
        """Test that subdirectories under /var, /tmp are safe (per spec)."""
        # Issue spec: only root and home should be blocked
        # .env.example lists /var/www/myproject as safe example
        assert is_dangerous_path("/var/www/myproject") is False
        assert is_dangerous_path("/tmp/build") is False
        assert is_dangerous_path("/var/log/myapp") is False

    def test_is_dangerous_path_system_directory_itself_unsafe(self):
        """Test that system directories themselves are still dangerous."""
        # /var itself, /tmp itself should still be blocked
        assert is_dangerous_path("/var") is True
        assert is_dangerous_path("/tmp") is True
        assert is_dangerous_path("/etc") is True

    def test_is_dangerous_path_windows_root(self):
        """Test that Windows drive roots are detected as dangerous."""
        # Create a mock Path object that simulates Windows drive root behavior
        mock_path = MagicMock(spec=Path)
        mock_path.parent = mock_path  # Root characteristic

        with patch("pathlib.Path") as mock_path_class:
            # When Path("C:\\").resolve() is called, return our mock
            mock_path_class.return_value.resolve.return_value = mock_path
            mock_path_class.return_value.resolve.return_value.parent = mock_path

            # Patch home() to avoid errors
            mock_path_class.home.return_value.resolve.return_value = Path("/home/user")

            # Patch os.name to simulate Windows
            with patch("os.name", "nt"):
                result = is_dangerous_path("C:\\\\")
                assert result is True

    def test_is_dangerous_path_symlink_to_root(self, tmp_path):
        """Test that symlinks to root are detected as dangerous."""
        link = tmp_path / "link_to_root"
        try:
            link.symlink_to("/")
            assert is_dangerous_path(str(link)) is True
        except (OSError, NotImplementedError):
            # Skip on systems that don't support symlinks (e.g., Windows without admin)
            pass

    def test_is_dangerous_path_relative_escape(self):
        """Test that relative paths escaping to root are detected."""
        # Only test if we're not already at root
        if os.getcwd() != "/":
            # Try to escape to root via relative path
            escape_path = os.path.join(os.getcwd(), "../" * 10)
            # Resolve and check if it's root
            resolved = Path(escape_path).resolve()
            if resolved == Path("/"):
                assert is_dangerous_path(escape_path) is True

    def test_is_dangerous_path_system_directories(self):
        """Test that system directories are detected as dangerous."""
        # These should be dangerous on POSIX systems
        if os.name == "posix":
            assert is_dangerous_path("/etc") is True
            assert is_dangerous_path("/bin") is True
            assert is_dangerous_path("/usr/bin") is True
            assert is_dangerous_path("/sbin") is True
            assert is_dangerous_path("/tmp") is True
            assert is_dangerous_path("/var") is True

    def test_is_dangerous_path_safe_user_directories(self):
        """Test that user directories outside system paths are safe."""
        # Directories commonly used for user projects should be safe
        if os.name == "posix":
            # /opt is not in the system directories list
            assert is_dangerous_path("/opt/myapp") is False
            # /srv is not in the system directories list
            assert is_dangerous_path("/srv/web/project") is False

    def test_is_dangerous_path_invalid_path(self):
        """Test that invalid paths are treated as dangerous (fail-safe)."""
        # Non-existent path with problematic characters
        # The implementation should handle this gracefully
        result = is_dangerous_path("\x00invalid\x00path")
        # Should either raise or return True (fail-safe)
        assert isinstance(result, bool)
        # Explicitly check it's treated as dangerous
        assert result is True

    def test_is_dangerous_path_windows_system_directories(self):
        """Test that Windows system directories are detected as dangerous."""
        # This test documents expected behavior on Windows
        # Currently returns False (not implemented), should return True after fix
        with patch("multi_llm_chat.mcp.filesystem_server.os.name", "nt"):
            # Windows system directories should be blocked
            assert is_dangerous_path("C:\\Windows") is True
            assert is_dangerous_path("C:\\Program Files") is True

    def test_is_dangerous_path_tilde_expansion(self):
        """Test that ~ (tilde) is properly expanded to home directory."""
        # Single tilde should expand to home
        assert is_dangerous_path("~") is True
        # Tilde with subdirectory should be safe
        assert is_dangerous_path("~/projects/myapp") is False


class TestCreateFilesystemServerConfig:
    """Tests for filesystem server configuration factory."""

    def test_create_config_explicit_root(self, tmp_path):
        """Test creating config with explicit root directory."""
        # Note: tmp_path may be under /var on macOS, which is now blocked.
        # Use allow_dangerous=True for testing purposes.
        config = create_filesystem_server_config(str(tmp_path), allow_dangerous=True)
        assert config.name == "filesystem"
        # Path should be resolved to absolute
        assert str(tmp_path.resolve()) in config.server_args[1]

    def test_create_config_default_cwd(self):
        """Test creating config defaults to current working directory."""
        with patch("os.getcwd", return_value="/mock/cwd"):
            with patch("pathlib.Path.resolve") as mock_resolve:
                mock_resolve.return_value = Path("/mock/cwd")
                with patch("pathlib.Path.is_dir", return_value=True):
                    with patch(
                        "multi_llm_chat.mcp.filesystem_server.is_dangerous_path", return_value=False
                    ):
                        config = create_filesystem_server_config()
                        assert "/mock/cwd" in config.server_args[1]

    def test_create_config_rejects_dangerous_path_by_default(self):
        """Test that dangerous paths are rejected by default."""
        import pytest

        with pytest.raises(ValueError, match="SECURITY ERROR"):
            create_filesystem_server_config("/")

    def test_create_config_allows_dangerous_with_flag(self, caplog):
        """Test that dangerous paths can be allowed with explicit flag."""
        with caplog.at_level(logging.WARNING):
            # Need to mock is_dir to avoid actual filesystem check on /
            with patch("pathlib.Path.is_dir", return_value=True):
                config = create_filesystem_server_config("/", allow_dangerous=True)
                assert config.server_args == ["mcp-server-filesystem", "/"]
                # Should log warning even when allowed
                assert any("DANGEROUS PATH ALLOWED" in record.message for record in caplog.records)

    def test_create_config_rejects_nonexistent_path(self):
        """Test that non-existent paths are rejected."""
        import pytest

        nonexistent = "/nonexistent/path/that/does/not/exist"
        with pytest.raises(ValueError, match="does not exist"):
            create_filesystem_server_config(nonexistent)

    def test_create_config_rejects_file_path(self, tmp_path):
        """Test that file paths (not directories) are rejected."""
        import pytest

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            create_filesystem_server_config(str(test_file))

    def test_create_config_expands_tilde(self):
        """Test that ~ is properly expanded in path."""
        import pytest

        # Home directory itself is dangerous
        with pytest.raises(ValueError, match="SECURITY ERROR"):
            create_filesystem_server_config("~")

    def test_create_config_command(self, tmp_path):
        """Test that config uses uvx command."""
        config = create_filesystem_server_config(str(tmp_path), allow_dangerous=True)
        assert config.server_command == "uvx"
        assert "mcp-server-filesystem" in config.server_args

    def test_create_config_timeout(self, tmp_path):
        """Test that config uses default MCP timeout."""
        config = create_filesystem_server_config(str(tmp_path), allow_dangerous=True)
        assert config.timeout == 120

    def test_create_config_custom_timeout(self, tmp_path):
        """Test that config accepts custom timeout."""
        config = create_filesystem_server_config(str(tmp_path), timeout=300, allow_dangerous=True)
        assert config.timeout == 300
