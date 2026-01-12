# Multi-LLM Chat

## 概要
このツールは、GeminiとChatGPTの2つの大規模言語モデル（LLM）に対し、単一のインターフェースから指示を出し、会話文脈を共有しながら対話を行うためのものです。ユーザーは「司会者」として、メンション機能を用いて応答するAIを指定できます。

Web UI版とCLI版の2つのインターフェースを提供します。

## 機能
- **Web UI**: Gradioによる、チャット形式の使いやすいグラフィカルインターフェース。
- **CLI (REPL)**: 従来のコマンドラインによる対話型インターフェース。
- **統一会話履歴管理**: セッション中の全てのやり取りを時系列で保持します。
  - **Web UI**: ユーザーID別に履歴を保存・読み込み・管理できます。
  - **CLI**: `/history` コマンドで履歴操作が可能です。
- **メンション機能**: `@gemini`, `@chatgpt`, `@all` のメンションにより、特定のLLMまたは両方のLLMに応答をルーティングします。メンションなしの入力は思考メモとして履歴にのみ追加されます。
- **API連携**: Google Gemini APIおよびOpenAI ChatGPT APIとの連携をサポートします。
- **環境変数からのAPIキー読み込み**: APIキーは環境変数または`.env`ファイルから安全に読み込まれます。

## インストール

1. **リポジトリのクローン**:
   ```bash
   git clone <リポジトリのURL>
   cd multi-llm-chat/repo
   ```

2. **`uv`のインストール**:
   Pythonのパッケージ管理ツール`uv`をインストールします。詳細な手順は[uvの公式ドキュメント](https://docs.astral.sh/uv/installation/)を参照してください。
   例 (macOS/Linux):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **仮想環境の作成とアクティベート**:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

4. **依存関係のインストール**:
   ```bash
   uv sync --extra dev
   ```
   `pyproject.toml`と`uv.lock`に定義されたランタイム／テスト依存（`dev`エクストラを含む）がすべてインストールされます。依存関係を追加・更新した場合は`pyproject.toml`を編集した上で`uv lock`を実行し、続けて`uv sync --extra dev`で環境を最新化してください。

## 開発

このプロジェクトは**テスト駆動開発（TDD）**を採用しています。機能追加やバグ修正を行う際は、**必ずテストを先に書いてから実装**してください。詳細は `CONTRIBUTING.md` を参照してください。

### TDD ワークフロー

1. **Red**: 失敗するテストを書く
2. **Green**: テストを通す最小限の実装を追加
3. **Refactor**: テストを保ったままコードを改善

### 重要なドキュメント

- **[開発ワークフロー](doc/development_workflow.md)**: TDD、レビュー、CI/CDのガイドライン
- **[構造化コンテンツ移行計画](doc/migration_plan.md)**: 履歴データ形式の段階的移行ロードマップ
- **[アーキテクチャ](doc/architecture.md)**: 3層アーキテクチャの設計思想

### コード品質チェック

プロジェクトでは[Ruff](https://docs.astral.sh/ruff/)を使用して、コードのlintとフォーマットを行っています。

**Lintチェック**:
```bash
uv run ruff check .
```

**フォーマット適用**:
```bash
uv run ruff format .
```

**フォーマット確認のみ（CI用）**:
```bash
uv run ruff format --check .
```

### テストの実行

```bash
uv run pytest
```

すべてのPull Requestは、CI（GitHub Actions）でlintとテストが自動実行されます。ローカルでも同じチェックを実行してから、コミットすることを推奨します。

### Git フックの設定（推奨）

コミット前とプッシュ前に自動的にチェックを実行するGitフックをインストールできます。

```bash
sh hooks/install.sh
```

これにより以下のフックが有効になります:
- **pre-commit**: コミット前にRuffのlintとフォーマットチェックを実行
- **pre-push**: プッシュ前にpytestを実行

フックをスキップしたい場合は `git commit --no-verify` または `git push --no-verify` を使用してください。

## 使い方

### 1. APIキーの設定

プロジェクトのルートディレクトリに`.env`ファイルを作成し、以下の形式でAPIキーを設定します。
```
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```
`GOOGLE_API_KEY`は必須です。`OPENAI_API_KEY`はChatGPTを使用する場合に設定してください。

#### 履歴保存設定（オプション）
- `CHAT_HISTORY_DIR`: 履歴ファイルの保存先（デフォルト: `XDG_DATA_HOME/multi_llm_chat/chat_histories` または `~/.multi_llm_chat/chat_histories`）
- `CHAT_HISTORY_USER_ID`: CLI/非対話環境で利用するユーザーID（未設定時は起動時に入力を要求）

#### コンテキスト圧縮設定（オプション）

会話履歴が長くなった際のトークン数管理のため、以下の環境変数を設定できます。

```
# モデル別の最大コンテキスト長（トークン数）
GEMINI_MAX_CONTEXT_LENGTH=100000
CHATGPT_MAX_CONTEXT_LENGTH=50000

# デフォルトの最大コンテキスト長（未設定時: 4096）
DEFAULT_MAX_CONTEXT_LENGTH=4096

# トークン推定バッファファクター（未設定時: 1.2）
# 非OpenAIモデルの概算トークン数に適用される安全マージン
TOKEN_ESTIMATION_BUFFER_FACTOR=1.2
```

詳細は `.env.example` を参照してください。

### 2. Web UI版の実行 (推奨)

仮想環境がアクティベートされていることを確認し、以下のコマンドでWeb UIを起動します。
```bash
python app.py
```
ブラウザで `http://127.0.0.1:7860` を開いてください。

#### システムプロンプトの設定

Web UIの上部にある「System Prompt」入力欄で、AIの役割や応答スタイルを指定できます。例:
```
あなたは親切なプログラミングアシスタントです。
コードの説明は具体例を交えて行ってください。
```

トークン数が表示され、上限を超えると警告が表示されます。

#### 履歴管理機能

Web UIでは、ユーザーID別に会話履歴を保存・読み込み・管理できます。

##### ユーザーIDの入力

**⚠️ 重要**: ユーザーIDは認証機能ではありません。任意の文字列を入力できるため、他人のIDを使用すれば他人の履歴にアクセスできます。本番環境では認証機構の追加を検討してください。

1. Web UI左側の「User ID」欄に任意の識別子を入力します（例: `user001`）
2. ユーザーIDを入力しないと、履歴管理機能と送信ボタンは無効化されます

##### 履歴の保存

1. 「Save Name」欄に保存名を入力します（例: `meeting-notes-2024`）
2. 「Save History」ボタンをクリックします
3. 同名の履歴が既に存在する場合、上書き確認ダイアログが表示されます
   - **Yes**: 既存履歴を上書きします
   - **No**: 保存をキャンセルします

**注意**: 保存名に入力された半角英数字、アンダースコア(`_`)、ハイフン(`-`)以外の文字は、自動的にアンダースコアに変換されます。

##### 履歴の読み込み

1. 「Saved Histories」ドロップダウンから読み込みたい履歴を選択します
2. 「Load History」ボタンをクリックします
3. 未保存の変更がある場合、確認ダイアログが表示されます
   - **Yes**: 現在の会話を破棄して履歴を読み込みます
   - **No**: 読み込みをキャンセルします
4. 読み込まれた履歴の内容（チャット履歴とシステムプロンプト）が表示されます

##### 新規会話の開始

1. 「New Chat」ボタンをクリックします
2. 未保存の変更がある場合、確認ダイアログが表示されます
   - **Yes**: 現在の会話を破棄して新規会話を開始します
   - **No**: キャンセルします
3. チャット履歴とシステムプロンプトがクリアされます

##### 履歴ファイルの保存場所

履歴ファイルは以下のディレクトリに保存されます：
- **Linux/macOS**: `$XDG_DATA_HOME/multi_llm_chat/chat_histories/<user_id>/`、またはフォールバックとして `~/.multi_llm_chat/chat_histories/<user_id>/`
- **Windows**: `%USERPROFILE%/.multi_llm_chat/chat_histories/<user_id>/`

環境変数 `CHAT_HISTORY_DIR` で保存先ディレクトリをカスタマイズできます。

#### ローカルネットワークでの共有

同じネットワーク上のスマートフォンなど、他のデバイスからアクセスしたい場合は、以下のコマンドでアプリケーションを起動します。

```bash
MLC_SERVER_NAME=0.0.0.0 python app.py
```

### 3. CLI版の実行

コマンドラインで対話を行いたい場合は、`chat_logic.py`を実行します。
```bash
python chat_logic.py
```

#### CLIコマンド

プロンプト(`> `)が表示されたら、以下の形式で入力します。

##### メンション機能
- **Geminiに話しかける**: `@gemini こんにちは`
- **ChatGPTに話しかける**: `@chatgpt 自己紹介して`
- **両方に話しかける**: `@all 今日の天気は？`
- **思考メモ（API呼び出しなし）**: `これはメモです`

##### システムプロンプトコマンド
- **設定**: `/system あなたは親切なアシスタントです`
- **表示**: `/system`
- **クリア**: `/system clear`

##### 終了
- `exit` または `quit`

## アーキテクチャ

プロジェクトは責務分離された3層アーキテクチャを採用しています:

### モジュール構成

```
src/multi_llm_chat/
├── core.py          # 共通インターフェース（ファサード）
├── token_utils.py   # トークン計算ユーティリティ
├── history_utils.py # 履歴整形ユーティリティ
├── compression.py   # 履歴圧縮ロジック
├── validation.py    # 検証ロジック
├── llm_provider.py  # LLMプロバイダー抽象化
├── cli.py           # CLIインターフェース
├── webui.py         # Web UIインターフェース
├── app.py           # 互換性レイヤー
└── chat_logic.py    # 互換性レイヤー
```

詳細は[アーキテクチャ設計書](doc/architecture.md)を参照してください。

## 実装の優先順位と依存関係

機能開発は、以下の順序で進めることを推奨します。これは各仕様書間の依存関係を考慮したものです。  
並行度の違いで担当者を分ける場合は `issues/done/009-task-sequential-core-system-prompt.md`（フェーズ1: 完了）と `issues/010-task-parallel-context-history.md`（フェーズ2: 並行可能）を参照してください。

1.  ✅ **全体リファクタリングとシステムプロンプト機能 (`system_prompt_feature_requirements.md`)** - 完了
    -   `core.py`, `webui.py`, `cli.py`へのファイル分割リファクタリング
    -   システムプロンプト設定機能（Web UI / CLI）
    -   トークンカウント表示と上限チェック
    -   **Epic 009で実装完了**

2.  **コンテキスト圧縮機能 と 会話履歴機能**
    -   以下の2つの機能は、システムプロンプト機能の実装後に並行して開発可能です。
    -   ✅ **コンテキスト圧縮機能 (`context_compression_requirements.md`)** - 実装完了（Epic 10）
        -   ✅ Task A: トークンガードレール（モデル別最大コンテキスト長、トークン計算、スライディングウィンドウ枝刈り）
        -   ✅ Task B: UI/CLI統合（警告表示、自動枝刈り適用）
    -   ✅ **会話履歴の保存・管理機能 (`history_feature_requirements.md`)** - 実装完了（Epic 10）
        -   ✅ CLI: `/history list`, `/history save`, `/history load`, `/history new` 実装済み
        -   ✅ Web UI: ユーザーID別履歴管理パネル実装済み（保存・読込・新規・確認フロー）
    -   **統合テスト**: 3つの新機能は相互に影響するため、各機能の実装完了後、結合して動作を確認する統合テストを実施することが重要です。

## 開発ロードマップ
- [x] 実際のAPI呼び出しの実装
- [x] エラーハンドリングの強化
- [x] ストリーミング応答のサポート
- [x] GradioによるWeb UIの実装
- [x] テストの追加
- [x] **Epic 009: Coreリファクタリングとシステムプロンプト機能** ✅ 完了
  - 3層アーキテクチャへの移行（`core.py`, `cli.py`, `webui.py`）
  - システムプロンプト設定機能（UI/CLI）
  - トークンカウント表示と上限チェック
  - 後方互換性レイヤーの実装
- [ ] Epic 004: コンテキスト圧縮とトークンガードレール（`issues/004-epic-context-compression.md`）
  - [x] Task A: コアロジック実装（トークン計算、スライディングウィンドウ枝刈り）
  - [ ] Task B: UI/CLI統合
- [ ] Epic 005: 会話履歴の永続化と管理（`issues/005-epic-history-management.md`）
- [ ] 設定ファイルの外部化

これらのEpicを担当者単位に細分化する場合は、フェーズ1（完了済み）は`issues/done/009-task-sequential-core-system-prompt.md`を、フェーズ2（並行可能）は`issues/010-task-parallel-context-history.md`の補助タスク票を参照してください。

## ライセンス
[LICENSE](LICENSE) (TBD)
