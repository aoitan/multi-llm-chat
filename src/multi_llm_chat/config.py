"""Application configuration repository.

Centralizes access to configuration values loaded from environment variables
or other sources. Provides a clean interface for all application layers.

This module implements the Repository pattern for configuration management,
decoupling business logic from environment variable access.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration container.

    This dataclass holds all configuration values used throughout the application.
    Values are typically loaded from environment variables during initialization.
    """

    # API Keys
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Model settings
    gemini_model: str = "models/gemini-pro-latest"
    chatgpt_model: str = "gpt-4o"

    # Token settings
    token_buffer_factor: float = 1.2
    token_buffer_factor_with_tools: float = 1.5

    # MCP settings
    mcp_enabled: bool = False
    mcp_timeout_seconds: int = 120
    mcp_filesystem_root: Optional[str] = None

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings.

        Returns:
            list[str]: List of warning messages for missing or invalid configuration.
        """
        issues = []

        if not self.google_api_key:
            issues.append("GOOGLE_API_KEY not set - Gemini features will be unavailable")

        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY not set - ChatGPT features will be unavailable")

        if self.token_buffer_factor <= 0:
            issues.append(f"Invalid TOKEN_BUFFER_FACTOR: {self.token_buffer_factor}")

        if self.token_buffer_factor_with_tools <= 0:
            issues.append(
                f"Invalid TOKEN_BUFFER_FACTOR_WITH_TOOLS: {self.token_buffer_factor_with_tools}"
            )

        if self.mcp_timeout_seconds <= 0:
            issues.append(f"Invalid MCP_TIMEOUT_SECONDS: {self.mcp_timeout_seconds}")

        return issues


# Global configuration instance (set once at startup)
_config: Optional[AppConfig] = None


def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables.

    This should be called once during application initialization
    (typically from init_runtime()).

    Returns:
        AppConfig: Configuration instance populated from environment variables.
    """

    def _get_env_float(key: str, default: float) -> float:
        """Safely parse float from environment variable with fallback."""
        val_str = os.getenv(key)
        if val_str is None:
            return default
        try:
            return float(val_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for {key}: '{val_str}'. Using default value: {default}.")
            return default

    def _get_env_int(key: str, default: int) -> int:
        """Safely parse int from environment variable with fallback."""
        val_str = os.getenv(key)
        if val_str is None:
            return default
        try:
            return int(val_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for {key}: '{val_str}'. Using default value: {default}.")
            return default

    mcp_enabled_str = os.getenv("MULTI_LLM_CHAT_MCP_ENABLED", "false").lower()
    mcp_enabled = mcp_enabled_str in ("true", "1", "yes")

    mcp_filesystem_root = os.getenv("MCP_FILESYSTEM_ROOT")
    if mcp_filesystem_root is None:
        mcp_filesystem_root = os.getcwd()

    config = AppConfig(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "models/gemini-pro-latest"),
        chatgpt_model=os.getenv("CHATGPT_MODEL", "gpt-4o"),
        token_buffer_factor=_get_env_float("TOKEN_BUFFER_FACTOR", 1.2),
        token_buffer_factor_with_tools=_get_env_float("TOKEN_BUFFER_FACTOR_WITH_TOOLS", 1.5),
        mcp_enabled=mcp_enabled,
        mcp_timeout_seconds=_get_env_int("MCP_TIMEOUT_SECONDS", 120),
        mcp_filesystem_root=mcp_filesystem_root,
    )

    # Log validation issues
    issues = config.validate()
    for issue in issues:
        logger.warning(issue)

    return config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance.

    This should only be called once during application initialization.

    Args:
        config: AppConfig instance to use globally.

    Raises:
        RuntimeError: If configuration has already been set.
    """
    global _config
    if _config is not None:
        raise RuntimeError("Configuration already set. Call reset_config() first.")
    _config = config
    logger.debug("Configuration initialized")


def get_config() -> AppConfig:
    """Get the global configuration instance.

    Returns:
        AppConfig: The global configuration instance.

    Raises:
        RuntimeError: If configuration has not been initialized.
                     Call init_runtime() first.
    """
    if _config is None:
        raise RuntimeError(
            "Configuration not initialized. Call init_runtime() at application startup."
        )
    return _config


def reset_config() -> None:
    """Reset configuration state.

    This function is intended for testing purposes only.
    It allows tests to reset the configuration between test cases.
    """
    global _config
    _config = None


def is_config_initialized() -> bool:
    """Check if configuration has been initialized.

    Returns:
        bool: True if configuration is initialized, False otherwise.
    """
    return _config is not None
