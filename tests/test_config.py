"""Tests for configuration repository module."""

import os
from unittest.mock import patch

import pytest

from multi_llm_chat.config import (
    AppConfig,
    get_config,
    is_config_initialized,
    load_config_from_env,
    reset_config,
    set_config,
)


@pytest.fixture(autouse=True)
def reset_config_state():
    """Reset configuration state before each test."""
    reset_config()
    yield
    reset_config()


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_appconfig_defaults(self):
        """Test that AppConfig has correct default values."""
        config = AppConfig()
        assert config.google_api_key is None
        assert config.openai_api_key is None
        assert config.gemini_model == "models/gemini-pro-latest"
        assert config.chatgpt_model == "gpt-3.5-turbo"
        assert config.token_buffer_factor == 1.2
        assert config.token_buffer_factor_with_tools == 1.5
        assert config.mcp_enabled is False
        assert config.mcp_timeout_seconds == 120

    def test_appconfig_validation_no_api_keys(self):
        """Test validation warns about missing API keys."""
        config = AppConfig()
        issues = config.validate()
        assert len(issues) == 2
        assert any("GOOGLE_API_KEY" in issue for issue in issues)
        assert any("OPENAI_API_KEY" in issue for issue in issues)

    def test_appconfig_validation_with_api_keys(self):
        """Test validation passes with API keys set."""
        config = AppConfig(google_api_key="test-key", openai_api_key="test-key")
        issues = config.validate()
        assert len(issues) == 0

    def test_appconfig_validation_invalid_buffer_factor(self):
        """Test validation catches invalid buffer factors."""
        config = AppConfig(token_buffer_factor=0)
        issues = config.validate()
        assert any("TOKEN_BUFFER_FACTOR" in issue for issue in issues)

    def test_appconfig_validation_invalid_timeout(self):
        """Test validation catches invalid timeout values."""
        config = AppConfig(mcp_timeout_seconds=-1)
        issues = config.validate()
        assert any("MCP_TIMEOUT_SECONDS" in issue for issue in issues)


class TestConfigRepository:
    """Tests for configuration repository functions."""

    def test_is_config_initialized_before_set(self):
        """Test that configuration is not initialized before set_config()."""
        assert not is_config_initialized()

    def test_is_config_initialized_after_set(self):
        """Test that configuration is initialized after set_config()."""
        config = AppConfig()
        set_config(config)
        assert is_config_initialized()

    def test_set_config_stores_instance(self):
        """Test that set_config() stores the configuration instance."""
        config = AppConfig(google_api_key="test-key")
        set_config(config)
        retrieved = get_config()
        assert retrieved is config
        assert retrieved.google_api_key == "test-key"

    def test_set_config_twice_raises_error(self):
        """Test that calling set_config() twice raises RuntimeError."""
        config1 = AppConfig()
        config2 = AppConfig()
        set_config(config1)
        with pytest.raises(RuntimeError, match="Configuration already set"):
            set_config(config2)

    def test_get_config_before_init_raises_error(self):
        """Test that get_config() raises RuntimeError before initialization."""
        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            get_config()

    def test_reset_config_clears_state(self):
        """Test that reset_config() clears the configuration state."""
        config = AppConfig()
        set_config(config)
        assert is_config_initialized()
        reset_config()
        assert not is_config_initialized()

    def test_reset_config_allows_reinit(self):
        """Test that reset_config() allows setting a new configuration."""
        config1 = AppConfig(google_api_key="key1")
        set_config(config1)
        reset_config()
        config2 = AppConfig(google_api_key="key2")
        set_config(config2)
        assert get_config().google_api_key == "key2"


class TestLoadConfigFromEnv:
    """Tests for load_config_from_env() function."""

    def test_load_config_from_env_defaults(self):
        """Test loading configuration with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config_from_env()
            assert config.google_api_key is None
            assert config.openai_api_key is None
            assert config.gemini_model == "models/gemini-pro-latest"
            assert config.chatgpt_model == "gpt-3.5-turbo"
            assert config.token_buffer_factor == 1.2
            assert config.token_buffer_factor_with_tools == 1.5
            assert config.mcp_enabled is False
            assert config.mcp_timeout_seconds == 120

    def test_load_config_from_env_with_keys(self):
        """Test loading configuration with API keys set."""
        env = {
            "GOOGLE_API_KEY": "test-google-key",
            "OPENAI_API_KEY": "test-openai-key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
            assert config.google_api_key == "test-google-key"
            assert config.openai_api_key == "test-openai-key"

    def test_load_config_from_env_with_custom_models(self):
        """Test loading configuration with custom model names."""
        env = {
            "GEMINI_MODEL": "models/gemini-custom",
            "CHATGPT_MODEL": "gpt-5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
            assert config.gemini_model == "models/gemini-custom"
            assert config.chatgpt_model == "gpt-5"

    def test_load_config_from_env_with_custom_buffer_factors(self):
        """Test loading configuration with custom buffer factors."""
        env = {
            "TOKEN_BUFFER_FACTOR": "1.5",
            "TOKEN_BUFFER_FACTOR_WITH_TOOLS": "2.0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
            assert config.token_buffer_factor == 1.5
            assert config.token_buffer_factor_with_tools == 2.0

    def test_load_config_from_env_with_custom_timeout(self):
        """Test loading configuration with custom MCP timeout."""
        env = {"MCP_TIMEOUT_SECONDS": "300"}
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
            assert config.mcp_timeout_seconds == 300

    def test_load_config_from_env_logs_warnings(self, caplog):
        """Test that load_config_from_env() logs validation warnings."""
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level("WARNING"):
                load_config_from_env()
                assert any("GOOGLE_API_KEY" in record.message for record in caplog.records)
                assert any("OPENAI_API_KEY" in record.message for record in caplog.records)

    def test_load_config_from_env_mcp_enabled_true(self):
        """Test loading configuration with MCP enabled."""
        for value in ["true", "True", "1", "yes", "YES"]:
            with patch.dict(os.environ, {"MULTI_LLM_CHAT_MCP_ENABLED": value}, clear=True):
                config = load_config_from_env()
                assert config.mcp_enabled is True, f"Failed for value: {value}"

    def test_load_config_from_env_mcp_enabled_false(self):
        """Test loading configuration with MCP disabled."""
        for value in ["false", "False", "0", "no", "NO", ""]:
            with patch.dict(os.environ, {"MULTI_LLM_CHAT_MCP_ENABLED": value}, clear=True):
                config = load_config_from_env()
                assert config.mcp_enabled is False, f"Failed for value: {value}"

    def test_load_config_from_env_invalid_float_value(self):
        """Test that invalid float values fall back to defaults with warning."""
        with patch.dict(os.environ, {"TOKEN_BUFFER_FACTOR": "invalid"}, clear=True):
            config = load_config_from_env()
            # Should fall back to default
            assert config.token_buffer_factor == 1.2

    def test_load_config_from_env_invalid_int_value(self):
        """Test that invalid int values fall back to defaults with warning."""
        with patch.dict(os.environ, {"MCP_TIMEOUT_SECONDS": "not_a_number"}, clear=True):
            config = load_config_from_env()
            # Should fall back to default
            assert config.mcp_timeout_seconds == 120

    def test_load_config_from_env_empty_string_values(self):
        """Test that empty string values fall back to defaults."""
        env = {
            "TOKEN_BUFFER_FACTOR": "",
            "TOKEN_BUFFER_FACTOR_WITH_TOOLS": "",
            "MCP_TIMEOUT_SECONDS": "",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
            assert config.token_buffer_factor == 1.2
            assert config.token_buffer_factor_with_tools == 1.5
            assert config.mcp_timeout_seconds == 120
