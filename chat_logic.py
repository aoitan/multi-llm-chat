# Initialize runtime environment (load .env) before importing other modules
from multi_llm_chat.runtime import init_runtime

init_runtime()

from multi_llm_chat.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
