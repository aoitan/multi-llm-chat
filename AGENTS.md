# Repository Guidelines

## Project Structure & Module Organization
Root scripts separate the two user experiences: `app.py` hosts the Gradio UI (web), while `chat_logic.py` implements the CLI loop plus API adapters for Gemini and ChatGPT. Supporting assets live under `doc/` (feature specs) and `issues/` (planning notes). Tests reside in `tests/` and currently focus on CLI behavior via `tests/test_chat_logic.py`. Keep new modules colocated with their interface (e.g., future `core.py` next to the UI/CLI entry points) so both front-ends can import without circular dependencies.

## Build, Test, and Development Commands
Use `uv` for Python environment setup:
```bash
uv venv .venv && source .venv/bin/activate
uv pip install google-generativeai openai python-dotenv gradio pytest tiktoken
```
Run the Web UI with `python app.py` (or `MLC_SERVER_NAME=0.0.0.0 python app.py` to share on a LAN). Launch the CLI with `python chat_logic.py`. Execute the regression suite via `pytest` from the repo root; tests mock API calls, so no keys are needed.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions, and UPPER_CASE for config constants such as `GEMINI_MODEL`. Keep streaming helpers pure where possible and prefer small, testable functions (e.g., `format_history_for_gemini`). UI patches that work around upstream bugs—like the runtime Gradio monkey patch in `app.py`—should include concise comments explaining the rationale and version. Type hints are welcome when they aid readability, but ergonomic generator code may remain unannotated.

## Testing Guidelines
All behavioral changes require a `tests/test_*.py` addition or update. Mirror existing naming (`test_<feature>`) and rely on `unittest.mock.patch` for I/O or API boundaries so tests stay hermetic. For new LLM flows, assert both routing (which API is called) and history shape (roles/content). Aim to keep `pytest` passing locally before opening a PR; add regression cases whenever a bug is fixed.

## Commit & Pull Request Guidelines
Commits follow the `<scope>: <short summary>` pattern already in history (e.g., `doc: update specs`). Keep scope tokens short (`feat`, `fix`, `refactor`, `tests`). Pull requests should describe the user goal, outline key changes, note any follow-up todos, and link related issues or specs under `doc/`. Include verification steps (`pytest`, manual CLI/Web walkthroughs) plus screenshots/GIFs when the UI changes.

## Security & Configuration Tips
Store secrets in `.env`; `GOOGLE_API_KEY` is mandatory, while `OPENAI_API_KEY` unlocks ChatGPT routing. Never commit `.env` or paste raw keys in logs. When debugging API errors, prefer the safe helper `list_gemini_models()` instead of printing keys, and document any new environment variables in `README.md`.
