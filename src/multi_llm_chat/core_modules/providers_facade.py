"""Provider access facade.

This module provides:
1. Provider factory and access (re-exports from llm_provider)
2. Debug utilities for Gemini models

All functions in this module delegate to llm_provider or google.generativeai.
"""

import logging

import google.generativeai as genai

from ..llm_provider import GOOGLE_API_KEY

logger = logging.getLogger(__name__)


def list_gemini_models(verbose: bool = True) -> list:
    """List available Gemini models (for debugging).

    Args:
        verbose: If True, print models to console. Default: True.

    Returns:
        List of model names that support generateContent.
    """
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment variables or .env file.")
        return []

    genai.configure(api_key=GOOGLE_API_KEY)
    models = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            models.append(m.name)
            logger.debug("Available Gemini model: %s", m.name)

    if verbose:
        print("利用可能なGeminiモデル:")
        for name in models:
            print(f"  - {name}")

    logger.info("Found %d Gemini models", len(models))
    return models
