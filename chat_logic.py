# Initialize runtime environment (load .env) before importing other modules
import warnings

from multi_llm_chat.runtime import init_runtime

init_runtime()

# Deprecated in v1.X, will be removed in v2.0.0 (Issue #116)
warnings.warn(
    "Running via 'python chat_logic.py' is deprecated. "
    "Use 'python -m multi_llm_chat.cli' instead. "
    "This script will be removed in v2.0.0",
    DeprecationWarning,
    stacklevel=2,
)

from multi_llm_chat.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
