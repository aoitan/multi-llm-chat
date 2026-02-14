# Initialize runtime environment (load .env) before importing other modules
import sys
import warnings

from multi_llm_chat.runtime import init_runtime

init_runtime()

# Deprecated in v1.X, will be removed in v2.0.0 (Issue #116)
# Use FutureWarning to ensure visibility to end users
warnings.warn(
    "Running via 'python app.py' is deprecated. "
    "Use 'python -m multi_llm_chat.webui' instead. "
    "This script will be removed in v2.0.0",
    FutureWarning,
    stacklevel=2,
)
# Also print to stderr for immediate visibility
print(
    "DeprecationWarning: Running via 'python app.py' is deprecated. "
    "Use 'python -m multi_llm_chat.webui' instead. "
    "This script will be removed in v2.0.0",
    file=sys.stderr,
)

from multi_llm_chat.webui import demo, launch  # noqa: E402

__all__ = ["demo", "launch"]

if __name__ == "__main__":
    launch()
