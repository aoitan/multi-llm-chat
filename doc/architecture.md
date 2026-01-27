# アーキテクチャ設計書

## 概要

Multi-LLM Chatは、複数のLLM（Gemini、ChatGPT）との対話を統一的に管理するPythonアプリケーションです。Epic 009のリファクタリングにより、責務分離された3層アーキテクチャを採用しました。

## アーキテクチャ全体図

```mermaid
graph TB
    subgraph UI["ユーザーインターフェース層"]
        WebUI["webui.py<br/>Gradio UI<br/>- Chat UI<br/>- System Prompt<br/>- Token Display<br/>- History Panel"]
        CLI["cli.py<br/>REPL Loop<br/>- /system<br/>- @mentions<br/>- Commands"]
    end
    
    subgraph Service["ビジネスロジック層 (Epic 017)"]
        ChatService["ChatService (chat_logic.py)<br/>- parse_mention()<br/>- process_message()<br/>- LLMルーティング<br/>- 履歴管理"]
    end
    
    subgraph Core["コアロジック層 (低レベルAPI)"]
        CorePy["core.py (Facade)"]
        TokenUtils["token_utils.py<br/>- トークン計算<br/>- get_token_info()"]
        HistoryUtils["history_utils.py<br/>- 履歴整形<br/>- 定数定義"]
        Compression["compression.py<br/>- 履歴圧縮<br/>- スライディングウィンドウ"]
        Validation["validation.py<br/>- バリデーション"]
        LLMProvider["llm_provider.py<br/>- 戦略パターン<br/>- API呼び出し"]
    end
    
    subgraph Persistence["永続化層 (Epic 017)"]
        HistoryStore["HistoryStore (history.py)<br/>- save_history()<br/>- load_history()<br/>- list_histories()"]
    end
    
    subgraph External["外部サービス層"]
        Gemini["Google Gemini API"]
        ChatGPT["OpenAI ChatGPT API"]
        FileSystem["File System<br/>(chat_histories/)"]
    end
    
    subgraph Compat["後方互換性レイヤー（オプション）"]
        AppPy["app.py → webui へ再エクスポート"]
        ChatLogic["chat_logic.py → 再エクスポート:<br/>- ChatService (新)<br/>- core.* (旧)<br/>- main() (旧CLI)"]
    end
    
    WebUI --> ChatService
    CLI --> ChatService
    WebUI --> HistoryStore
    ChatService --> CorePy
    CorePy --> TokenUtils
    CorePy --> HistoryUtils
    CorePy --> Compression
    CorePy --> Validation
    CorePy --> LLMProvider
    LLMProvider --> TokenUtils
    Compression --> TokenUtils
    Validation --> TokenUtils
    LLMProvider --> Gemini
    LLMProvider --> ChatGPT
    HistoryStore --> FileSystem
    AppPy -.-> WebUI
    ChatLogic -.-> ChatService
    ChatLogic -.-> CorePy
    ChatLogic -.-> CLI
```

## モジュール設計

### 1. コアロジック層

#### 1.1 `src/multi_llm_chat/core.py`

**責務**: 共通インターフェース（ファサード）。各機能別ユーティリティを統合し、後方互換性を提供。

#### 1.2 `src/multi_llm_chat/token_utils.py`

**責務**: トークン計算ロジック、コンテキスト長情報の管理。

#### 1.3 `src/multi_llm_chat/history_utils.py`

**責務**: 履歴データの整形、プロバイダー判定、共通定数（`LLM_ROLES`など）の定義。

#### 1.4 `src/multi_llm_chat/compression.py`

**責務**: 履歴データのスライディングウィンドウによる圧縮・枝刈り。

#### 1.5 `src/multi_llm_chat/validation.py`

**責務**: システムプロンプトやコンテキスト長の検証。

#### 1.6 `src/multi_llm_chat/llm_provider.py`

**責務**: 各LLMプロバイダーへのルーティング層。後方互換性を維持しつつ、`providers/`パッケージへ委譲。

#### 1.7 `src/multi_llm_chat/providers/`

**責務**: LLMプロバイダー実装の独立化（戦略パターン）。

- `base.py`: `LLMProvider`抽象基底クラス（共通インターフェース定義）
- `gemini.py`: Google Gemini実装（`GeminiProvider`, `GeminiToolCallAssembler`クラス）
- `openai.py`: OpenAI/ChatGPT実装（`ChatGPTProvider`, `OpenAIToolCallAssembler`クラス）

### 2. ビジネスロジック層 (Epic 017追加)

**責務**: ビジネスロジック層 - メンション解析、LLMルーティング、履歴管理

##### `ChatService`クラス

チャットセッションを管理し、UI層（CLI/WebUI）からビジネスロジックを分離するサービスクラス。

| 機能カテゴリ | メソッド | 説明 |
|------------|---------|------|
| **初期化** | `__init__(display_history, logic_history, system_prompt)` | セッション状態を保持（display: UI用、logic: API用） |
| **メンション解析** | `parse_mention(message)` | `@gemini`, `@chatgpt`, `@all`を検出（モジュール関数） |
| **メッセージ処理** | `process_message(user_message)` | メンション解析→LLM呼び出し→履歴更新（ジェネレータ） |
| **LLM呼び出し** | `_process_gemini(message)`, `_process_chatgpt(message)` | 各LLMへのAPI呼び出しとストリーミング処理 |

##### 設計の特徴

- **UI形式の分離**: 
  - `display_history`: UI表示用（Gradio Chatbot形式: `[[user, assistant], ...]`）
  - `logic_history`: API呼び出し用（`[{"role": "user", "content": "..."}, ...]`）
- **`@all`処理**: 
  - 履歴スナップショットを作成し、GeminiとChatGPTに同一の文脈を提供
  - 両者の応答を順次（Gemini→ChatGPT）追加
- **ストリーミング対応**: `yield`で中間状態を返し、リアルタイム表示を実現
- **エラーハンドリング**: メンション欠落時は早期リターン（CLIではメモとして処理）

##### 後方互換性レイヤー

`chat_logic.py`は以下の再エクスポートも提供:
- `main()`: 旧CLI実装（`cli.py`への移行推奨）
- `core.*`: `core.py`の全関数（既存コードとの互換性維持）

### 2. ユーザーインターフェース層

#### 2.1 Web UI: `src/multi_llm_chat/webui.py`

**責務**: Gradio UIの実装

##### 主要機能

| コンポーネント | 関数/変数 | 説明 |
|-------------|----------|------|
| **UI関数** | `update_token_display(system_prompt, logic_history, model_name)` | トークン数表示を更新 |
| | `check_send_button_with_user_id(user_id, system_prompt, logic_history, model_name)` | 送信ボタンの有効/無効を判定（user_idとトークン制限を両方チェック） |
| | `check_history_buttons_enabled(user_id)` | 履歴管理ボタンの有効/無効を判定（戻り値: dict of `gr.update()`） |
| | `respond(user_message, display_history, logic_history, system_prompt, user_id)` | チャット応答の中心関数（user_id必須） |
| **起動関数** | `launch(server_name=None, debug=True)` | Gradioアプリを起動 |
| **UI要素** | `demo` | Gradio Blocksインスタンス |
| | `user_id_input` | ユーザーID入力欄（履歴管理用） |
| | `system_prompt_input` | システムプロンプト入力欄 |
| | `token_display` | トークンカウント表示 |
| | `chatbot_ui` | チャット表示エリア |
| | `save_history_btn`, `load_history_btn`, `new_chat_btn` | 履歴管理ボタン |

##### 特殊処理
- **Gradioバグ回避**: JSON Schema内のbool型処理のためのモンキーパッチ適用
- **user_id検証**: `respond()`関数内でuser_idの空チェックを実施し、未入力時はエラーメッセージを返す

#### 2.2 CLI: `src/multi_llm_chat/cli.py`

**責務**: コマンドラインインターフェースの実装

##### 主要機能

| 機能 | 関数/変数 | 説明 |
|------|----------|------|
| **メインループ** | `main()` | REPLループ、`(history, system_prompt)`を返す |
| **コマンド処理** | `_handle_system_command(args, system_prompt, current_model)` | `/system`コマンドの処理 |
| **応答処理** | `_process_response_stream(stream, model_name)` | ストリーミング応答の表示 |
| **ユーティリティ** | `_clone_history(history)` | 履歴の浅いコピーを作成 |

##### サポートするコマンド

| コマンド | 説明 |
|---------|------|
| `/system <prompt>` | システムプロンプトを設定 |
| `/system` | 現在のシステムプロンプトを表示 |
| `/system clear` | システムプロンプトをクリア |
| `exit` / `quit` | CLIを終了 |

##### メンション機能

| メンション | 動作 |
|-----------|------|
| `@gemini <message>` | Geminiに送信 |
| `@chatgpt <message>` | ChatGPTに送信 |
| `@all <message>` | 両方に送信（履歴は分岐前のスナップショット） |
| メンションなし | 思考メモとして履歴に追加のみ |

### 3. 互換性レイヤー（オプション）

#### 3.1 `src/multi_llm_chat/app.py`

旧インターフェースを維持するため、`webui`モジュールの`demo`, `launch`, `respond`を再エクスポート。

```python
from .webui import demo, launch, respond
__all__ = ["demo", "launch", "respond"]
```

#### 3.2 `src/multi_llm_chat/chat_logic.py`

旧インターフェースを維持するため、`core`と`cli`モジュールの関数を再エクスポート。

```python
from .cli import main
from .core import (
    CHATGPT_MODEL, GEMINI_MODEL, GOOGLE_API_KEY, OPENAI_API_KEY,
    call_chatgpt_api, call_gemini_api,
    format_history_for_chatgpt, format_history_for_gemini,
    list_gemini_models,
)
```

### 4. エントリーポイント

#### ルートレベルのラッパー

| ファイル | インポート元 | 用途 |
|---------|------------|------|
| `app.py` | `multi_llm_chat.webui` | Web UI起動 (`python app.py`) |
| `chat_logic.py` | `multi_llm_chat.cli` | CLI起動 (`python chat_logic.py`) |

## データフロー

### システムプロンプト適用フロー

```mermaid
sequenceDiagram
    actor User
    participant UI as UI Layer<br/>(webui/cli)
    participant Prep as core.py<br/>prepare_request()
    participant API as core.py<br/>call_*_api()
    participant Ext as External API
    
    User->>UI: システムプロンプト入力
    UI->>Prep: system_prompt引数として渡す
    alt OpenAI
        Prep->>API: [{"role":"system", "content":...}, ...history]
    else Gemini
        Prep->>API: (system_prompt, history) tuple
    end
    API->>Ext: API呼び出し
    Ext-->>API: レスポンス
    API-->>User: ストリーミング表示
```

### トークンカウントフロー

```mermaid
sequenceDiagram
    actor User
    participant UI as UI Layer
    participant Token as core.py<br/>get_token_info()
    
    User->>UI: システムプロンプト変更
    UI->>Token: get_token_info(text, model_name)
    Token-->>UI: {token_count, max_context_length, is_estimated}
    UI->>UI: 表示更新
    UI->>UI: 送信ボタン有効/無効判定
    UI-->>User: 結果表示
```

## テスト構造

### テストファイルとモジュールの対応

| テストファイル | 対象モジュール | テスト数 |
|-------------|-------------|---------|
| `tests/test_core.py` | `src/multi_llm_chat/core.py` | 10 |
| `tests/test_cli.py` | `src/multi_llm_chat/cli.py` | 8 |
| `tests/test_webui.py` | `src/multi_llm_chat/webui.py` | 20 |
| `tests/test_chat_logic.py` | 互換性レイヤー | 3 |

### テスト方針

- **ユニットテスト**: `unittest.mock.patch`でAPI呼び出しをモック
- **統合テスト**: 実際のAPI呼び出しは行わない（hermetic tests）
- **カバレッジ**: 全ての公開関数をテスト
- **TDD**: Red-Green-Refactorサイクルを厳守

## 設定とシークレット管理

### 環境変数

| 変数名 | 必須 | デフォルト | 説明 |
|-------|-----|----------|------|
| `GOOGLE_API_KEY` | ✅ | なし | Gemini API Key |
| `OPENAI_API_KEY` | ❌ | なし | OpenAI API Key |
| `GEMINI_MODEL` | ❌ | `models/gemini-pro-latest` | 使用するGeminiモデル |
| `CHATGPT_MODEL` | ❌ | `gpt-3.5-turbo` | 使用するChatGPTモデル |
| `MLC_SERVER_NAME` | ❌ | `127.0.0.1` | Web UIのホスト名 |

### `.env`ファイル例

```bash
GOOGLE_API_KEY="your-gemini-key-here"
OPENAI_API_KEY="your-openai-key-here"
GEMINI_MODEL="models/gemini-2.0-flash-exp"
CHATGPT_MODEL="gpt-4o"
```

## パフォーマンス考慮事項

### キャッシング戦略

- **APIクライアント**: モジュールレベルでシングルトンとしてキャッシュ
- **Geminiモデル**: `_gemini_model`グローバル変数
- **OpenAIクライアント**: `_openai_client`グローバル変数

### ストリーミング処理

- 両APIとも`stream=True`でストリーミング応答
- ジェネレーターパターンで逐次出力
- UIレスポンスの向上

## 拡張性

### 新しいLLMの追加方法

1. **core.py**: 
   - `call_<model>_api()`関数を追加
   - `format_history_for_<model>()`関数を追加
   - `prepare_request()`に分岐を追加

2. **cli.py**: 
   - メンションパターンに`@<model>`を追加
   - 応答処理に分岐を追加

3. **webui.py**: 
   - 必要に応じてUI要素を追加

4. **テスト**: 
   - 各モジュールに対応するテストを追加

### 将来の機能追加

以下の機能は既存アーキテクチャで実装可能:

- ✅ **システムプロンプト**: 実装済み
- 🔲 **コンテキスト圧縮**: `core.py`に追加予定
- 🔲 **履歴永続化**: `core.py`に追加予定
- 🔲 **複数モデル同時利用**: 現在の`@all`を拡張

## コードメトリクス

### モジュールサイズ

| モジュール | 行数 | 主要機能数 |
|-----------|-----|----------|
| `core.py` | 165 | 11 |
| `cli.py` | 122 | 4 |
| `webui.py` | 185 | 5 |
| `app.py` (互換) | 4 | - |
| `chat_logic.py` (互換) | 26 | - |

### リファクタリング効果

- **削減行数**: 336行（旧実装から）
- **正味削減**: 306行
- **コード重複**: ほぼゼロ（共通処理はcoreに集約）

## 参考ドキュメント

- [システムプロンプト機能要件](./doc/system_prompt_feature_requirements.md)
- [Epic 009タスク票](./issues/009-task-sequential-core-system-prompt.md)
- [コンテキスト圧縮要件](./doc/context_compression_requirements.md)
- [履歴管理要件](./doc/history_feature_requirements.md)
