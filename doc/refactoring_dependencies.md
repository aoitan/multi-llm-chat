# core.py リファクタリング: 関数依存関係分析

このドキュメントは Issue #103 のリファクタリングにおいて、各関数の移動可否と依存関係を記録します。

## 移動対象関数の依存関係

### 1. `load_api_key(env_var_name)` (L101-103)
- **現在の場所**: `core.py`
- **移動先**: `llm_provider.py`
- **依存**: `os.getenv()` のみ（標準ライブラリ）
- **被参照**: core.py内では未使用、外部から直接呼び出し可能性あり
- **移動戦略**: `llm_provider.py` に移動し、`core.py` からは再エクスポート

### 2. `format_history_for_gemini(history)` (L106-112)
- **現在の場所**: `core.py`
- **実装**: `GeminiProvider.format_history()` へ委譲済み（DEPRECATED）
- **移動先**: 不要（既にwrapper）
- **依存**: `GeminiProvider`
- **被参照**: テストコード、外部APIとして使用される可能性
- **移動戦略**: そのまま残す（既に委譲済みで3行のみ）

### 3. `format_history_for_chatgpt(history)` (L115-121)
- **現在の場所**: `core.py`
- **実装**: `ChatGPTProvider.format_history()` へ委譲済み（DEPRECATED）
- **移動先**: 不要（既にwrapper）
- **依存**: `ChatGPTProvider`
- **被参照**: テストコード、外部APIとして使用される可能性
- **移動戦略**: そのまま残す（既に委譲済みで3行のみ）

### 4. `extract_text_from_chunk(chunk, model_name)` (L266-280)
- **現在の場所**: `core.py`
- **移動先**: `history_utils.py`（テキスト抽出はhistory処理に関連）
- **依存**: 
  - `_get_provider_name_from_model()` from `history_utils`
  - 標準ライブラリのみ（dict/list操作）
- **被参照**: `stream_text_events()` (L248)、テストコード
- **移動戦略**: `history_utils.py` に移動し、`core.py` からは再エクスポート

## 移動不要な関数（既に委譲済み）

以下は既に他モジュールへ委譲済みで、core.pyには薄いwrapperのみ残っている:

- `prepare_request()` → `history_utils._prepare_request()`
- `_estimate_tokens()` → `token_utils._estimate_tokens_impl()`
- `get_max_context_length()` → `token_utils._get_max_context_length()`
- `prune_history_sliding_window()` → `compression._prune_history_sliding_window()`
- `get_pruning_info()` → `compression._get_pruning_info()`
- `validate_system_prompt_length()` → `validation._validate_system_prompt_length()`
- `validate_context_length()` → `validation._validate_context_length()`

これらは公開APIとして維持し、内部実装は既存モジュールに任せる。

## 残留する関数（オーケストレーション層）

以下はcore.pyの中核機能として残す:

- `AgenticLoopResult` - データクラス
- `call_gemini_api()` - Gemini API呼び出しオーケストレーション
- `call_chatgpt_api()` - ChatGPT API呼び出しオーケストレーション
- `stream_text_events()` - ストリーミング制御
- `execute_with_tools_stream()` - Agentic Loopのストリーミング版
- `execute_with_tools()` - Agentic Loopのメイン実装
- `execute_with_tools_sync()` - Agentic Loopの同期版
- `calculate_tokens()` - トークン計算のエントリーポイント
- `get_token_info()` - DEPRECATED wrapper（互換性のため残す）
- `list_gemini_models()` - デバッグ用ユーティリティ（CLIへ移動検討）

## リファクタリング順序

計画に従い、以下の順序で実施:

1. ✅ **Commit 1**: テスト用フィクスチャ抽出（完了）
2. ✅ **Commit 2**: このドキュメント作成（完了）
3. **Commit 3**: `history_utils.py` に移動先セクション準備
4. **Commit 4**: `extract_text_from_chunk()` を移動
5. **Commit 5**: `load_api_key()` を移動
6. **Commit 6**: `list_gemini_models()` をCLIへ移動（オプション）

## 注意事項

- `format_history_for_gemini/chatgpt()` は既にProvider層へ委譲済みのため、移動不要
- 計画のCommit 4-5は統合され、Commit 4のみ実施（extract_text_from_chunk）
- 公開APIのシグネチャは維持し、内部で委譲することで互換性を保つ
