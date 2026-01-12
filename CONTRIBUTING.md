# Contributing

## Development Workflow

This project follows **Test-Driven Development (TDD)**. All feature additions and bug fixes must follow the Red-Green-Refactor cycle.

### Quick Start
1. Read `AGENTS.md` for project structure, coding style, and detailed guidelines
2. Set up your environment: `uv venv .venv && source .venv/bin/activate && uv sync --extra dev`
3. Install Git hooks (recommended): `sh hooks/install.sh`
4. Before coding, write failing tests first
5. Run tests: `uv run pytest`
6. Run linter: `uv run ruff check . && uv run ruff format .`

### TDD Cycle (Required)

#### Red Phase
- Write a failing test in `tests/test_*.py` that describes the desired behavior
- Run `uv run pytest` to confirm it fails for the right reason

#### Green Phase
- Write minimal code to make the test pass
- Run `uv run pytest` to confirm all tests pass

#### Refactor Phase
- Clean up code while keeping tests green
- Run `uv run pytest` after each change

### Git Hooks (Recommended)

This project provides Git hooks to catch issues before CI:
- **pre-commit**: Runs `ruff check` and `ruff format --check`
- **pre-push**: Runs `pytest` to ensure all tests pass

Install them with:
```bash
sh hooks/install.sh
```

To bypass hooks when necessary, use `--no-verify`:
```bash
git commit --no-verify
git push --no-verify
```

### Pull Request Requirements
- List all test cases added in the PR description
- Ensure `uv run ruff check .` and `uv run pytest` pass
- Include verification steps (manual testing if applicable)
- If TDD was skipped (emergency hotfix only), explicitly state why

For detailed guidelines, coding style, and project structure, refer to `AGENTS.md`.
