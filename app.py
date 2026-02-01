# Initialize runtime environment (load .env) before importing other modules
from multi_llm_chat.runtime import init_runtime

init_runtime()

from multi_llm_chat.webui import demo, launch  # noqa: E402

__all__ = ["demo", "launch"]

if __name__ == "__main__":
    launch()
