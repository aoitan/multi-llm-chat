"""Provider access facade.

This module provides:
1. Provider factory and access (re-exports from llm_provider)
2. Debug utilities for Gemini models

All functions in this module delegate to llm_provider or the appropriate Gemini SDK.
"""

import logging
import os
import warnings

# Dynamic import based on SDK selection (Issue #138)
# Support backward compatibility: USE_NEW_GEMINI_SDK is deprecated but still works
use_legacy_sdk = os.getenv("USE_LEGACY_GEMINI_SDK", "0") == "1"
use_new_sdk_deprecated = os.getenv("USE_NEW_GEMINI_SDK", "0") == "1"

if use_new_sdk_deprecated:
    warnings.warn(
        "USE_NEW_GEMINI_SDK is deprecated. The new SDK is now the default. "
        "To use legacy SDK, set USE_LEGACY_GEMINI_SDK=1 instead.",
        DeprecationWarning,
        stacklevel=2,
    )

if use_legacy_sdk:
    import google.generativeai as genai
else:
    import google.genai as genai

from ..config import get_config

logger = logging.getLogger(__name__)


def list_gemini_models(verbose: bool = True) -> list:
    """List available Gemini models (for debugging).

    Args:
        verbose: If True, print models to console. Default: True.

    Returns:
        List of model names that support generateContent.
        Note: For new SDK, filtering by supported_actions may not be available
        on all model types, so we include all returned models.
    """
    config = get_config()
    if not config.google_api_key:
        logger.error("GOOGLE_API_KEY not configured. Call init_runtime() first.")
        return []

    models = []
    use_legacy = os.getenv("USE_LEGACY_GEMINI_SDK", "0") == "1"

    try:
        if use_legacy:
            # Legacy SDK (google.generativeai)
            genai.configure(api_key=config.google_api_key)
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    models.append(m.name)
                    logger.debug("Available Gemini model: %s", m.name)
        else:
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
