# Task: Implement Ruff lint + GitHub Actions CI pipeline

## 概要
`doc/ci_lint_spec.md` で定義した lint/formatter（Ruff）と CI パイプライン（lint + pytest）をリポジトリに実装し、品質検証を自動化する。

## 作業詳細
- `pyproject.toml` の `[project.optional-dependencies].dev` に `ruff>=0.4` を追加し、`[tool.ruff]` / `[tool.ruff.lint]` の設定（line-length 100、対象ディレクトリ、`select=["E","F","B","I"]` 等）を記述する。
- `uv lock` でロックファイルを更新し、`uv sync --extra dev` の手順が Ruff を含むことを確認する。
- `.github/workflows/ci.yml` を新規作成し、仕様書の 2 ジョブ（`lint` と `tests`）を定義する。
  - `lint`: Python 3.10、`uv run ruff check .` と `uv run ruff format --check .`。
  - `tests`: Python 3.10/3.11 マトリクスで `uv run pytest`。
- `README.md` にローカル開発手順として `uv run ruff check .` / `ruff format` を追記し、lint を必須手順として周知する（エージェント向けの `AGENTS.md` では、CI で Ruff が必須であることを明文化する）。
- 初回 Ruff 実行で検出される違反を修正（必要に応じて `# noqa`/`# pragma: no cover` を最小限追加）。

## 完了条件
- `pyproject.toml` に Ruff の依存と設定が追加され、`uv lock` が更新されている。
- `.github/workflows/ci.yml` が存在し、ローカルで `act` もしくはリポジトリ CI 上で lint/tests の両ジョブが成功する。
- `README.md` 等で Ruff コマンドが案内されており、開発者が CI と同じチェックをローカル再現できる。
- `ruff check .` および `pytest` がリポジトリ HEAD でパスする。
