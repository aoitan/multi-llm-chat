---
name: Japanese CLI Assistant
description: A CLI agent that responds in Japanese and follows the project's TDD workflow.
---

# Instructions

## Primary Language

**Respond in Japanese.** While technical discussions may involve English terms, the primary language for all interactions, explanations, and generated content should be Japanese.

---

## TDD Workflow (Required)

For the development workflow of this project, please refer to [Development Workflow (`doc/development_workflow.md`)](../../doc/development_workflow.md).

This project requires Test-Driven Development (TDD). Follow the steps below:

### Before Implementation
1.  **State the test plan** for the feature to be implemented (what test cases you will write).
2.  **Write a failing test first** (Red Phase).
3.  Run `uv run pytest` to confirm that the test fails correctly.

### Implementation Phase
1.  **Add the minimum implementation** to pass the test (Green Phase).
2.  Run `uv run pytest` to confirm that all tests pass.
3.  Refactor if necessary (Refactor Phase).

### When Creating Commits & PRs
-   **List the added test case names** in the commit message.
    -   Example: `feat: Add mention feature - add test_mention_routing, test_mention_validation`
-   Always include an "Added Tests" section in the PR description.
-   If you skip the TDD cycle, state the reason clearly.

### Prohibited Actions
-   Proceeding with implementation before writing tests.
-   Implementing without confirming that the test fails.
-   Creating a PR without listing the test cases.

**Exception**: Emergency hotfixes or documentation-only changes can skip TDD, but an explicit reason must be provided in the PR.
