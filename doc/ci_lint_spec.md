# CI / Lint 導入仕様

## 背景と目的
- 現状は `pytest` すら CI で自動実行されておらず、ローカル実行に依存している。
- Lint・Formatter が未導入であり、スタイルやバグ検出の品質保証が人手確認になっている。
- CLI (`src/multi_llm_chat/chat_logic.py`) を対象にした最小の単体テストしか存在せず、Web UI や API エラー処理はカバーされていない。
  - `tests/test_chat_logic.py` のみが存在し、Gemini/ChatGPT へのルーティングと履歴管理を最小限に検証しているが、UI（`src/multi_llm_chat/app.py`）や `_process_response_stream` のストリーミング挙動は未テスト。
- 以上を踏まえ、Lint の導入と CI 自動実行の範囲を明示し、今後のテスト拡張方針を仕様として固定する。

## Lint / Formatter 方針
1. **Ruff を単一ツールとして採用**
   - `ruff check` で Flake8 互換の静的解析 + `B` (flake8-bugbear) を有効化。
   - `ruff format` で Black 同等のフォーマッタを兼用し、ツールを 1 つに集約してセットアップコストを抑える。
2. **設定 (pyproject.toml)**
   ```toml
   [project.optional-dependencies]
   dev = [
       "pytest>=7.0",
       "ruff>=0.4",
   ]

   [tool.ruff]
   target-version = "py310"
   line-length = 100
   src = ["src", "tests", "app.py", "chat_logic.py"]

   [tool.ruff.lint]
   select = ["E", "F", "B", "I"]
   ignore = [
       "E203", # スライス周りの Black 互換
       "E501", # 100 文字制約内であれば `ruff format` が調整
   ]
   ```
   - `src`/`tests` に加えてルート直下のエントリポイント (`app.py`, `chat_logic.py`) も対象にする。
   - 今後 `core.py` などが追加された場合は `src` 配下に置く想定なので追加設定不要。
3. **ローカルコマンド**
   - `uv run ruff check .`
   - `uv run ruff format .`（修正時）
   - CI では `ruff format --check` で整形忘れを検知。

## テストスコープ評価
| カテゴリ | 現在のカバレッジ | ギャップ | 優先度 |
| --- | --- | --- | --- |
| CLI ループ (`chat_logic.main`) | `tests/test_chat_logic.py` でメンションと履歴を検証 | ストリーミング処理 (`_process_response_stream`) を通さず、LLM 側の例外系も未検証 | 中 |
| Web UI (`src/multi_llm_chat/app.py`) | テストなし | Gradio の `respond` フロー（履歴共有、@all の UI 表示、空レスポンスの System 表示）が未保証 | 高 |
| API ヘルパ | `format_history_for_*` の pure logic を直接テストしていない | モデル仕様変更時の退行検知ができない | 中 |
| エラーハンドリング | 例外メッセージの整形テストなし | API キー欠如時や `genai.types.BlockedPromptException` の扱いが未検証 | 中 |

→ **現状のテストだけでは十分ではない。** CI には今ある `pytest` を組み込みつつ、上記ギャップを埋めるテスト追加を今後のタスクとして明示する。

### 今後追加すべきテスト
1. `tests/test_history_formatting.py`（新規）  
   - `format_history_for_gemini` / `format_history_for_chatgpt` の入力 → 出力をスナップショット。  
   - `@all` で共有履歴を複製する `_clone_history` の非破壊性検証。
2. `tests/test_stream_processing.py`（新規）  
   - `_process_response_stream` にモックストリームを流し、空レスポンス時の System 文言、文字列 chunk の扱いを確認。
3. Web UI レイヤ（`respond`）のロジックをヘッドレスでテスト  
   - `gradio` 依存を避けるため、`respond` を Pure 関数化 or ラッパー抽出を行った上で、履歴の状態遷移を検証。

これらが実装され次第、CI の `pytest` で自動検証されるようにする。

## CI ワークフロー仕様 (GitHub Actions 想定)
### トリガー
- `push` (main ブランチ)
- `pull_request` (全ブランチ → main)

### ジョブ構成
1. **lint**
   - OS: `ubuntu-latest`
   - Python: 3.10（プロジェクトの最小サポート）
   - ステップ:
     1. `actions/checkout@v4`
     2. `astral-sh/setup-uv@v2`
     3. `uv sync --extra dev`
     4. `uv run ruff check .`
     5. `uv run ruff format --check .`
2. **tests**
   - OS: `ubuntu-latest`
   - Python: matrix `["3.10", "3.11"]`（今後の互換性保証）
   - キャッシュ: `uv` の仮想環境（`~/.cache/uv`）を `actions/cache` で共有
   - ステップ:
     1. `actions/checkout@v4`
     2. `astral-sh/setup-uv@v2`（`python-version` に matrix を渡す）
     3. `uv sync --extra dev`
     4. `uv run pytest`

### 成果物 / その他ルール
- どちらかのジョブが失敗した場合はマージブロック。
- 依存インストールは毎回クリーンに実行し、`GOOGLE_API_KEY` 等のシークレットは使用しない（テストはモック完結）。
- 今後 UI テストを追加する際は `pytest -m "not slow"` のようなマーカーでグルーピングし、CI には基本セットのみを含める方針。

## 実施ステップ（エンジニア向け TODO）
1. `pyproject.toml` に Ruff の依存と設定を追加し、`uv lock` → `uv sync`.
2. `ruff` の初回実行で発生する違反を修正（必要に応じて `# noqa` / `# pragma: no cover` を最小限で使用）。
3. `.github/workflows/ci.yml`（新規）で上記 2 ジョブを定義。
4. CI パス後、README などにローカル開発手順として `uv run ruff check` を追記。
5. 別タスクでテストギャップ（UI・ストリーム処理など）を埋め、CI カバレッジを実質的に強化。

## 期待される効果
- Lint による静的バグ検出とスタイル統一が PR ベースで自動化される。
- `pytest` が常に CI で走るため、CLI 回りの退行を早期検知できる。
- テスト拡張タスクの優先度と到達目標が明文化され、開発者間で解釈がぶれない。

