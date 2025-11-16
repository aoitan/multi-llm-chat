# Task: Manage dependencies with `pyproject.toml` + `uv.lock`

## 概要
依存関係管理を`pyproject.toml`と`uv.lock`へ移行し、`uv sync`で再現性の高いセットアップを提供する。

## 作業詳細
- 既存の`uv pip install ...`手順を置き換えるため、`pyproject.toml`にパッケージ一覧と開発依存（例: `pytest`, `tiktoken`）を定義する。
- `uv lock`を実行して`uv.lock`を生成し、リポジトリに追加する。
- `AGENTS.md`や`README.md`のセットアップ手順を`uv sync`ベースに更新し、`uv.lock`の使い方と更新方針を明記する。
- 必要に応じて`pyproject.toml`内でPythonバージョンやツール設定（pytest, lintなど）を指定する。

## 完了条件
- `pyproject.toml`と`uv.lock`がリポジトリに追加され、`uv sync`のみで開発環境が再現できる。
- ドキュメント（`AGENTS.md`, `README.md`）が新しいセットアップ手順に合わせて更新されている。
- 既存の`uv pip install ...`手順が撤廃され、`uv sync`の利用が明確に案内されている。
