# MCP Filesystem統合ガイド

## 概要

Multi-LLM Chatは、MCP (Model Context Protocol) を通じて、LLMがローカルファイルシステムにアクセスできる機能を提供します。この機能により、LLMはプロジェクトのファイルを読み取り、その内容を基に回答を生成できます。

**主な用途**:
- 「README.mdを読んで要約して」のような指示が可能
- コードベース内のファイル構造を把握
- 特定のファイルの内容を基にした質問・回答

**重要な前提**:
- **読み取り系ツール想定**: 現在公開されているツール（read_file, list_directory等）は読み取り用途を想定していますが、将来的にmcp-server-filesystemのバージョンや設定により書き込み系ツールが追加される可能性があります。安全のため、重要なファイルが含まれないディレクトリを指定してください
- **セキュリティ**: 指定したディレクトリ配下のみを対象とするよう構成されていますが、最終的な挙動は `mcp-server-filesystem` のバージョンや設定に依存します

## セットアップ

### 1. MCPサーバーのインストール

filesystemサーバーは`uvx`経由で自動的に起動されます。追加のインストールは不要です。

### 2. 環境変数の設定

`.env`ファイルに以下の設定を追加します：

```bash
# MCP機能を有効化
MULTI_LLM_CHAT_MCP_ENABLED=true

# アクセスを許可するディレクトリ（省略可）
# 省略時はカレントディレクトリ（os.getcwd()）が使用されます
MCP_FILESYSTEM_ROOT=/path/to/your/project
```

#### セキュリティ保護

以下のディレクトリは**危険パス**としてデフォルトでブロックされます：

**❌ ブロックされるパス**:
- システムルート: `/` または `C:\` (Windows)
- ホームディレクトリ: `~`, `/home/username`, `/Users/username`
- POSIXシステムディレクトリ: `/etc`, `/bin`, `/sbin`, `/usr/bin`, `/usr/sbin`, `/boot`, `/dev`, `/proc`, `/sys`, `/tmp`, `/var`
- Windowsシステムディレクトリ: `C:\Windows`, `C:\Windows\System32`, `C:\Windows\SysWOW64`, `C:\Program Files`, `C:\Program Files (x86)`, `C:\ProgramData`

**✅ 安全なパスの例**:
```bash
# プロジェクト専用ディレクトリ
MCP_FILESYSTEM_ROOT=/home/user/projects/myapp

# ワークスペース配下の特定ディレクトリ
MCP_FILESYSTEM_ROOT=/var/www/myproject

# 専用のMCPサンドボックス
MCP_FILESYSTEM_ROOT=/opt/mcp-workspaces/project1
```

**注意**: サブディレクトリは安全として扱われます。例えば、`/var/www/myproject`は許可されますが、`/var`自体は拒否されます。

#### 危険パス保護の無効化（非推奨）

完全に理解した上で危険パス保護を無効化する場合：

```bash
# ⚠️ セキュリティリスクを理解した上でのみ設定
MCP_ALLOW_DANGEROUS_PATHS=true
```

### 3. アプリケーションの起動

#### Web UI版
```bash
python app.py
```

#### CLI版
```bash
python chat_logic.py
```

どちらのインターフェースでもMCP機能は自動的に有効化されます。

## 使用例

### 基本的な使い方

```
You: @gemini README.mdを読んで内容を要約して

[Tool Call: read_file]
  Args: {'path': 'README.md'}
[Tool Result: read_file]
  # Multi-LLM Chat
  
  ## 概要
  このツールは、GeminiとChatGPTの2つの...

Gemini: README.mdの内容を確認しました。このプロジェクトは...
```

### ファイル一覧の取得

```
You: @gemini src/ディレクトリ配下のファイル一覧を教えて

[Tool Call: list_directory]
  Args: {'path': 'src/'}
[Tool Result: list_directory]
  - multi_llm_chat/
    - core.py
    - cli.py
    - webui.py
    ...

Gemini: src/ディレクトリには以下のファイルがあります...
```

### 複数ファイルの比較

```
You: @gemini core.pyとcli.pyの役割の違いを説明して

[Tool Call: read_file]
  Args: {'path': 'src/multi_llm_chat/core.py'}
[Tool Result: read_file]
  ...

[Tool Call: read_file]
  Args: {'path': 'src/multi_llm_chat/cli.py'}
[Tool Result: read_file]
  ...

Gemini: core.pyは...一方、cli.pyは...
```

## 利用可能なツール

MCPサーバーは以下のツールを提供します：

### `read_file`
- **説明**: 指定したパスのファイルを読み取る
- **引数**:
  - `path` (string): 読み取るファイルのパス（相対パスまたは絶対パス）
- **戻り値**: ファイルの内容（テキスト）

### `list_directory`
- **説明**: 指定したディレクトリ内のファイル・ディレクトリ一覧を取得
- **引数**:
  - `path` (string): 一覧を取得するディレクトリのパス
- **戻り値**: ファイル・ディレクトリ名の配列

### `search_files`
- **説明**: 正規表現パターンでファイル名を検索
- **引数**:
  - `path` (string): 検索を開始するディレクトリ
  - `pattern` (string): 検索パターン（正規表現）
- **戻り値**: マッチしたファイルパスの配列

### `get_file_info`
- **説明**: ファイルのメタ情報を取得
- **引数**:
  - `path` (string): 情報を取得するファイルのパス
- **戻り値**: ファイルサイズ、更新日時などのメタ情報

## トラブルシューティング

### MCP機能が動作しない

**症状**: LLMがファイルを読み取れない、ツール呼び出しが発生しない

**確認事項**:
1. `.env`ファイルで`MULTI_LLM_CHAT_MCP_ENABLED=true`が設定されているか
2. `MCP_FILESYSTEM_ROOT`が存在するディレクトリを指しているか
3. ログに`Initializing MCP servers...`や`All MCP servers started successfully`のようなメッセージが出力されているか

**デバッグ方法**:
```bash
# 起動時のログを確認
python app.py 2>&1 | grep -i mcp
```

### 危険パスのエラー

**症状**: アプリケーション起動時に失敗し、スタックトレースとともに  
`ValueError: [SECURITY ERROR] ... Cannot expose dangerous path ...`  
のようなエラーメッセージが表示される

**原因**: セキュリティ保護により、システムルートやホームディレクトリなどの危険なパスを`MCP_FILESYSTEM_ROOT`として公開しようとしています。

**解決策**:
1. プロジェクト専用のサブディレクトリを`MCP_FILESYSTEM_ROOT`に設定する
2. どうしても必要な場合は`MCP_ALLOW_DANGEROUS_PATHS=true`を設定して起動する（非推奨・本番環境では禁止推奨）

### ツール呼び出しタイムアウト

**症状**: `TimeoutError: Execution exceeded timeout of ... seconds`エラー

**原因**: 大きなファイルの読み取りや複雑な検索でタイムアウトが発生

**解決策**:
- より具体的な指示を与える（「全ファイルを読んで」→「README.mdを読んで」）
- ファイルサイズを事前に確認する

## セキュリティ上の注意

### アクセス範囲の制限

- **最小権限の原則**: 必要最小限のディレクトリのみを`MCP_FILESYSTEM_ROOT`に設定してください
- **機密情報**: APIキー、パスワード、トークンなどを含むファイルには注意が必要です

### 公開環境での使用

- Web UIを公開する場合、認証機能の追加を強く推奨します
- `MCP_FILESYSTEM_ROOT`が公開されても問題ないディレクトリであることを確認してください

### 監視とログ

- MCPツールの呼び出しはログに記録されます
- 定期的にログを確認し、予期しないアクセスがないかチェックしてください

## 今後の拡張予定

以下の機能は将来のバージョンで実装予定です：

- **書き込み機能**: ファイルの作成・編集機能（明示的な許可が必要）
- **複数ルート**: 複数のディレクトリを同時に公開
- **カスタムMCPサーバー**: filesystem以外のMCPサーバーのサポート
- **アクセス制御**: ファイル単位・ディレクトリ単位の細かい権限設定

## 参考リンク

- [MCP公式ドキュメント](https://modelcontextprotocol.io/)
- [mcp-server-filesystemリポジトリ](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
- [プロジェクトのアーキテクチャドキュメント](./architecture.md)
