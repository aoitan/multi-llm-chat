# Repository Guidelines

## Project Structure & Module Organization
Root scripts separate the two user experiences: `app.py` hosts the Gradio UI (web), while `chat_logic.py` implements the CLI loop plus API adapters for Gemini and ChatGPT. These scripts now proxy into the `src/multi_llm_chat/` package, which contains the actual implementation. _Epic 003 (Core Refactor) renames these entry points to `webui.py` and `cli.py`; this guide reflects the current state and should be updated after the refactor ships._ Supporting assets live under `doc/` (feature specs) and `issues/` (planning notes). Tests reside in `tests/` and currently focus on CLI behavior via `tests/test_chat_logic.py`. Keep new modules colocated with their interface (e.g., future `core.py` next to the UI/CLI entry points) so both front-ends can import without circular dependencies.

## Build, Test, and Development Commands
Use `uv` for Python environment setup and dependency syncing:
```bash
uv venv .venv && source .venv/bin/activate
uv sync --extra dev
```
This installs everything defined in `pyproject.toml` / `uv.lock`, including the `dev` extra for tests and Ruff. When dependencies change, update `pyproject.toml`, refresh the lockfile via `uv lock`, and re-run `uv sync --extra dev`.
Run the Web UI with `python app.py` (or `MLC_SERVER_NAME=0.0.0.0 python app.py` to share on a LAN). Launch the CLI with `python chat_logic.py`. _After Epic 003 lands, use `python webui.py` / `python cli.py` instead._ 

**Linting and formatting**: The repository enforces Ruff checks in CI. Before committing, run:
```bash
uv run ruff check .       # Static analysis
uv run ruff format .      # Auto-format code
```

**Testing**: Execute the regression suite via `uv run pytest` from the repo root; tests mock API calls, so no keys are needed. CI runs both lint and pytest on all PRs—ensure both pass locally first.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions, and UPPER_CASE for config constants such as `GEMINI_MODEL`. The project uses Ruff (configured in `pyproject.toml`) for linting (`select = ["E", "F", "B", "I"]`) and formatting (Black-compatible, 100-char line length). Keep streaming helpers pure where possible and prefer small, testable functions (e.g., `format_history_for_gemini`). UI patches that work around upstream bugs—like the runtime Gradio monkey patch in `webui.py` (currently `app.py`)—should include concise comments explaining the rationale and version. Type hints are welcome when they aid readability, but ergonomic generator code may remain unannotated.

## Testing Guidelines

### TDD Workflow (Required)
This project follows **Test-Driven Development (TDD)** for all feature additions and bug fixes. Follow the Red-Green-Refactor cycle:

1. **Red**: Write a failing test first that captures the desired behavior
2. **Green**: Write minimal code to make the test pass
3. **Refactor**: Clean up the implementation while keeping tests green

**Process**:
- Before implementing any feature, write one or more failing tests in `tests/test_*.py`
- Run `uv run pytest` to confirm tests fail for the right reason
- Implement the minimal code to make tests pass
- Refactor if needed, ensuring tests remain green
- Commit with a message listing the test cases added (e.g., "feat: add X功能 - Added test_x_behavior, test_x_edge_case")

**Exceptions**: Emergency hotfixes or documentation-only changes may skip the TDD cycle, but must be explicitly noted in the PR description with justification.

### Test Requirements
All behavioral changes require a `tests/test_*.py` addition or update. Mirror existing naming (`test_<feature>`) and rely on `unittest.mock.patch` for I/O or API boundaries so tests stay hermetic. For new LLM flows, assert both routing (which API is called) and history shape (roles/content). Add regression cases whenever a bug is fixed. CI enforces both Ruff compliance and pytest success on every PR.

## Commit & Pull Request Guidelines
Commits follow the `<scope>: <short summary>` pattern already in history (e.g., `doc: update specs`). Keep scope tokens short (`feat`, `fix`, `refactor`, `tests`). 

Pull requests should:
- Describe the user goal and outline key changes
- **List all test cases added** (required for TDD compliance, e.g., "Added: test_mention_routing, test_history_validation")
- Note any follow-up todos and link related issues or specs under `doc/`
- Include verification steps (`uv run pytest`, manual CLI/Web walkthroughs) plus screenshots/GIFs when the UI changes
- Explicitly note if TDD cycle was skipped (e.g., "Hotfix: skipped TDD due to production outage")

## Security & Configuration Tips
Store secrets in `.env`; `GOOGLE_API_KEY` is mandatory, while `OPENAI_API_KEY` unlocks ChatGPT routing. Never commit `.env` or paste raw keys in logs. When debugging API errors, prefer the safe helper `list_gemini_models()` instead of printing keys, and document any new environment variables in `README.md`.
