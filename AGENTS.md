# Agent Instructions

**Primary Language**: Respond in Japanese (日本語で応答してください). While technical discussions may involve English terms, the primary language for all interactions, explanations, and generated content should be Japanese.

---

# Repository Guidelines

For the development workflow of this project, please refer to [Development Workflow (`doc/development_workflow.md`)](./doc/development_workflow.md).

## Project Structure & Module Organization

The project follows a 3-tier architecture with clear separation of concerns:

```
src/multi_llm_chat/
├── core.py          # Core logic layer (API calls, history management, token calculation)
├── chat_service.py  # Business logic layer (ChatService, parse_mention, ASSISTANT_LABELS)
├── cli.py           # CLI interface layer (REPL loop, command processing)
├── webui.py         # Web UI layer (Gradio interface)
├── app.py           # Backward compatibility layer (re-exports webui)
└── chat_logic.py    # Backward compatibility layer (DEPRECATED: use chat_service)
```

**Entry points** at the repository root:
- `app.py` → Launches Web UI via `python app.py`
- `chat_logic.py` → Launches CLI via `python chat_logic.py`

**Supporting assets**:
- `doc/` - Feature specifications and architecture documentation
- `issues/` - Planning notes and task tickets
- `tests/` - Test suite organized by module:
  - `test_core.py` - Core logic tests (21 tests)
  - `test_cli.py` - CLI interface tests (9 tests)
  - `test_webui.py` - Web UI tests (7 tests)
  - `test_chat_service.py` - ChatService business logic tests
  - `test_context_compression.py` - Context compression tests (26 tests)
  - `test_history_store.py` - History management tests (8 tests)
  - `test_chat_logic.py` - Backward compatibility tests

**Design principles**:
- Core layer is UI-agnostic (no `print()`/`input()` except for debugging utilities)
- ChatService layer encapsulates business logic (mention parsing, LLM routing, history management)
- Both UI layers (`cli.py`, `webui.py`) import and use `chat_service.py`
- Backward compatibility layers (`app.py`, `chat_logic.py`) ensure existing code doesn't break
- All modules are in `src/multi_llm_chat/` package to avoid circular dependencies

See [Architecture Documentation](doc/architecture.md) for detailed design.

## Build, Test, and Development Commands
Use `uv` for Python environment setup and dependency syncing:
```bash
uv venv .venv && source .venv/bin/activate
uv sync --extra dev
```
This installs everything defined in `pyproject.toml` / `uv.lock`, including the `dev` extra for tests and Ruff. When dependencies change, update `pyproject.toml`, refresh the lockfile via `uv lock`, and re-run `uv sync --extra dev`.

**Running the applications**:
- Web UI: `python app.py` (or `MLC_SERVER_NAME=0.0.0.0 python app.py` to share on a LAN)
- CLI: `python chat_logic.py`

**Linting and formatting**: The repository enforces Ruff checks in CI. Before committing, run:
```bash
uv run ruff check .       # Static analysis
uv run ruff format .      # Auto-format code
```

**Testing**: Execute the regression suite via `uv run pytest` from the repo root; tests mock API calls, so no keys are needed. CI runs both lint and pytest on all PRs—ensure both pass locally first.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions, and UPPER_CASE for config constants such as `GEMINI_MODEL`. The project uses Ruff (configured in `pyproject.toml`) for linting (`select = ["E", "F", "B", "I"]`) and formatting (Black-compatible, 100-char line length). 

**Code organization principles**:
- Keep `core.py` UI-agnostic (no `print()`/`input()` except for debugging utilities like `list_gemini_models()`)
- Prefer small, testable functions (e.g., `format_history_for_gemini()`)
- Use generator patterns for streaming responses
- Cache expensive resources (API clients) at module level

**UI-specific patches**: Workarounds for upstream bugs (like the Gradio JSON Schema patch in `webui.py`) should include concise comments explaining the rationale and affected version.

**Type hints**: Welcome when they aid readability, but ergonomic generator code may remain unannotated.

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
- Commit with a message listing the test cases added（例: "feat: X機能を追加 - test_x_behavior, test_x_edge_case を追加"）

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