"""Provider access facade.

This module provides:
1. Provider factory and access (re-exports from llm_provider)
2. Debug utilities for Gemini models

All functions in this module delegate to llm_provider or google.generativeai.
"""

import logging

import google.generativeai as genai

from ..config import get_config

logger = logging.getLogger(__name__)


def list_gemini_models(verbose: bool = True) -> list:
    """List available Gemini models (for debugging).

    Args:
        verbose: If True, print models to console. Default: True.

    Returns:
        List of model names that support generateContent.
    """
    config = get_config()
    if not config.google_api_key:
        logger.error("GOOGLE_API_KEY not configured. Call init_runtime() first.")
        return []

    genai.configure(api_key=config.google_api_key)
    models = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            models.append(m.name)
            logger.debug("Available Gemini model: %s", m.name)

    if verbose:
        logger.info("利用可能なGeminiモデル:")
        for name in models:
            logger.info("  - %s", name)

    logger.info("Found %d Gemini models", len(models))
    return models
