"""Tests for Gemini SDK Adapter layer

Issue: #136 (Phase 1 - Adapter層導入)
Issue: #139 (Phase 4 - Legacy SDK removed)
"""

import unittest

from multi_llm_chat.providers.gemini_adapter import GeminiSDKAdapter


class TestGeminiSDKAdapter(unittest.TestCase):
    """Test abstract base class"""

    def test_is_abstract(self):
        """GeminiSDKAdapter should be abstract"""
        with self.assertRaises(TypeError):
            GeminiSDKAdapter()


if __name__ == "__main__":
    unittest.main()
