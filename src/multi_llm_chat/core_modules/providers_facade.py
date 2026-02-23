"""Provider access facade.

This module provides:
1. Provider factory and access (re-exports from llm_provider)
2. Debug utilities for Gemini models

All functions in this module delegate to llm_provider or the appropriate Gemini SDK.
Phase 4 (Issue #139): Legacy SDK (google.generativeai) removed. Using google.genai only.
"""

import logging

import google.genai as genai

from ..config import get_config

logger = logging.getLogger(__name__)


def list_gemini_models(verbose: bool = True) -> list:
    """List available Gemini models (for debugging).

    Args:
        verbose: If True, print models to console. Default: True.

    Returns:
        List of model names that support generateContent.
        Note: For the new SDK, we attempt to filter for models that support the
        'GENERATE_CONTENT' action, but models without this attribute may also be included.
    """
    config = get_config()
    if not config.google_api_key:
        logger.error("GOOGLE_API_KEY not configured. Call init_runtime() first.")
        return []

    models = []

    try:
        # New SDK (google.genai)
        client = genai.Client(api_key=config.google_api_key)
        for m in client.models.list():
            # New SDK models have supported_actions field
            # Filter for models that support GENERATE_CONTENT action
            model_name = m.name if hasattr(m, "name") else str(m)

            # Check if model supports generateContent action
            if hasattr(m, "supported_actions") and m.supported_actions:
                if "GENERATE_CONTENT" not in m.supported_actions:
                    logger.debug("Skipping model (no GENERATE_CONTENT): %s", model_name)
                    continue

            models.append(model_name)
            logger.debug("Available Gemini model: %s", model_name)
    except Exception as e:
        logger.error("Failed to list Gemini models: %s", e)
        return []

    if verbose:
        logger.info("利用可能なGeminiモデル:")
        for name in models:
            logger.info("  - %s", name)

    logger.info("Found %d Gemini models", len(models))
    return models
