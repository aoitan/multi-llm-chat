# Issue #81: Agentic Loop（ツール実行ループ）実装 - 進捗管理

## 概要
LLMがツール呼び出しを行った際に、実際にMCPサーバーのツールを実行し、結果をLLMにフィードバックする「Agentic Loop」を実装する。

レビュー結果に基づき、以下の5つのPhaseに分割して実装する。

---

## Phase 1: 基盤修正（不変性の回復）🔴 Critical
**目的**: 既存機能を壊さず、アーキテクチャの健全性を回復

### Task 1.1: `AgenticLoopResult` の導入と不変性対応
- [ ] `AgenticLoopResult` データクラスを定義
  - [ ] `chunks: List[Dict]` - ストリーミングチャンク
  - [ ] `history_delta: List[Dict]` - 追加する履歴エントリ
  - [ ] `final_text: str` - 最終的なテキスト応答
  - [ ] `iterations_used: int` - 使用したイテレーション数
  - [ ] `timed_out: bool` - タイムアウトフラグ
  - [ ] `error: Optional[str]` - エラーメッセージ

- [ ] `execute_with_tools()` を不変性に対応
  - [ ] `history` 引数を読み取り専用として扱う
  - [ ] 内部で `deepcopy` して `working_history` を作成
  - [ ] `working_history` のみを変更
  - [ ] `AgenticLoopResult` オブジェクトを返す

- [ ] テストの追加
  - [ ] `test_result_immutability()` - 結果オブジェクトが不変
  - [ ] `test_history_not_mutated()` - 元の history が変更されない
  - [ ] `test_history_delta_contains_only_new_entries()` - delta が正しい

- [ ] 既存テストの更新
  - [ ] `test_execute_with_tools_single_iteration`
  - [ ] `test_execute_with_tools_max_iterations`
  - [ ] `test_execute_with_tools_timeout`
  - [ ] `test_execute_with_tools_tool_error`

### Task 1.2: 同期ラッパーの追加
- [ ] `execute_with_tools_sync()` 関数を実装
  - [ ] イベントループの存在チェック
  - [ ] 既存ループがある場合は `RuntimeError`
  - [ ] `asyncio.run()` で非同期関数を実行

- [ ] テストの追加
  - [ ] `test_sync_wrapper_works_in_sync_context()` - 同期環境で動作
  - [ ] `test_sync_wrapper_raises_in_async_context()` - 非同期環境でエラー

### Task 1.3: 呼び出し側の更新
- [ ] `ChatService` の更新
  - [ ] `result = execute_with_tools()` で結果取得
  - [ ] `history.extend(result.history_delta)` で明示的に更新

- [ ] CLI の更新
  - [ ] `execute_with_tools_sync()` を使用
  - [ ] 結果オブジェクトから履歴を更新

- [ ] WebUI の更新（Phase 3 で実施）

### Phase 1 完了条件
- [ ] 全既存テスト（261テスト）が通過
- [ ] 新規テスト（10件）が通過
- [ ] `execute_with_tools()` が history を変更しない（不変性確認）
- [ ] 同期ラッパーで既存コードと互換性維持
- [ ] パフォーマンス劣化 < 10%（ベンチマーク）

**PR**: `feature/81-phase1-immutability` → `main`

---

## Phase 2: 履歴スキーマの標準化 🟡 High
**目的**: `role: "tool"` を正式に標準化し、検証ロジックを統一

### Task 2.1: `history_utils.py` の拡張
- [ ] ロール定義の追加
  - [ ] `TOOL_ROLES = {"tool"}` を定義
  - [ ] `ALL_ROLES = LLM_ROLES | USER_ROLES | TOOL_ROLES` を定義

- [ ] `validate_history_entry()` 関数の実装
  - [ ] `role` が `ALL_ROLES` に含まれるかチェック
  - [ ] `role: "tool"` の場合、content が tool_result のみかチェック
  - [ ] 不正な場合は `ValueError`

- [ ] テストの追加
  - [ ] `test_validate_tool_role_valid()` - 正常な tool role
  - [ ] `test_validate_tool_role_invalid_content()` - 不正な content
  - [ ] `test_content_to_text_handles_tool_results()` - tool_result の文字列化

### Task 2.2: Provider の `format_history()` 更新
- [ ] `GeminiProvider.format_history()` の拡張
  - [ ] `role: "tool"` を `role: "function"` に変換
  - [ ] `tool_result` を `function_response` に変換

- [ ] `ChatGPTProvider.format_history()` の拡張
  - [ ] `role: "tool"` をそのまま使用
  - [ ] `tool_result` を OpenAI 形式に変換

- [ ] テストの追加
  - [ ] `test_gemini_format_history_with_tool_results()` - Gemini 対応
  - [ ] `test_chatgpt_format_history_with_tool_results()` - ChatGPT 対応

### Phase 2 完了条件
- [ ] 全既存テスト + Phase 1 テストが通過
- [ ] 新規テスト（5件）が通過
- [ ] `role: "tool"` が `history_utils.py` で検証される
- [ ] Gemini/ChatGPT 両方で正しく動作

**PR**: `feature/81-phase2-schema` → `main`

---

## Phase 3: MCP クライアント管理（WebUI対応）🟢 Medium
**目的**: マルチセッション環境での MCP クライアントのライフサイクル管理

### Task 3.1: MCP Manager の実装
- [ ] `src/multi_llm_chat/webui/mcp_manager.py` を作成
  - [ ] `MCPManager` クラスの実装
  - [ ] `_clients: Dict[str, MCPClient]` でセッション管理
  - [ ] `get_or_create_client(session_id, ...)` メソッド
  - [ ] `close_client(session_id)` メソッド
  - [ ] `close_all()` メソッド

- [ ] テストの追加
  - [ ] `test_mcp_manager_per_session_isolation()` - セッション分離
  - [ ] `test_mcp_manager_cleanup_on_close()` - クリーンアップ確認
  - [ ] `test_mcp_manager_close_all()` - 全クライアント終了

### Task 3.2: WebUI との統合
- [ ] `src/multi_llm_chat/webui/handlers.py` の更新
  - [ ] `respond_with_tools()` 関数で `MCPManager` を使用
  - [ ] Gradio の `session_id` を取得
  - [ ] セッションごとに MCP クライアントを分離

- [ ] `src/multi_llm_chat/webui/app.py` の更新
  - [ ] アプリ終了時に `MCPManager.close_all()` を呼び出し

- [ ] テストの追加
  - [ ] `test_webui_multi_session_tool_execution()` - マルチセッション
  - [ ] `test_mcp_manager_no_resource_leak()` - リソースリーク確認

### Phase 3 完了条件
- [ ] 全既存テスト + Phase 1-2 テストが通過
- [ ] 新規テスト（5件）が通過
- [ ] WebUI で複数ユーザーが同時にツールを使用可能
- [ ] リソースリークなし（メモリプロファイリング）

**PR**: `feature/81-phase3-mcp-manager` → `main`

---

## Phase 4: CLI 統合の簡素化 🔵 Low
**目的**: CLI での Agentic Loop 使用を簡単にする

### Task 4.1: CLI での同期ラッパー使用
- [ ] `src/multi_llm_chat/cli.py` の更新
  - [ ] `_process_with_tools()` 関数を実装
  - [ ] `execute_with_tools_sync()` を使用
  - [ ] 結果オブジェクトから履歴を更新
  - [ ] UI に tool_call/tool_result を表示

- [ ] MCP クライアント初期化
  - [ ] `main()` で MCP クライアントを初期化
  - [ ] 環境変数から設定を読み込み
  - [ ] 終了時にクリーンアップ

- [ ] テストの追加
  - [ ] `test_cli_with_tools_sync_wrapper()` - 同期ラッパー使用

### Phase 4 完了条件
- [ ] 全既存テスト + Phase 1-3 テストが通過
- [ ] 新規テスト（1件）が通過
- [ ] CLI で Agentic Loop が正常に動作

**PR**: `feature/81-phase4-cli` → `main`

---

## Phase 5: テストカバレッジの拡充 🟣 Low
**目的**: 新スキーマと並行処理のテストを追加

### Task 5.1: 追加テストケースの実装
- [ ] `tests/test_agentic_loop.py` の拡充
  - [ ] `test_parallel_execution()` - 複数ループの並行実行
  - [ ] `test_deepcopy_performance()` - パフォーマンス確認
  - [ ] `test_error_recovery()` - エラーからの回復

- [ ] `tests/test_history_utils.py` の拡充
  - [ ] `test_validate_all_roles()` - 全ロールの検証
  - [ ] `test_mixed_content_types()` - 混在コンテンツ

- [ ] `tests/test_chat_service.py` の拡充
  - [ ] `test_process_message_with_tool_role()` - tool role の処理

- [ ] `tests/test_webui.py` の拡充
  - [ ] `test_webui_multi_user_tool_execution()` - 複数ユーザー並行

### Task 5.2: 統合テスト
- [ ] エンドツーエンドテスト
  - [ ] CLI: ユーザー入力 → ツール実行 → 結果表示
  - [ ] WebUI: 複数セッション → 並行ツール実行 → 分離確認

- [ ] パフォーマンステスト
  - [ ] `deepcopy` のオーバーヘッド測定
  - [ ] MCP Manager のスケーラビリティ確認

### Phase 5 完了条件
- [ ] 全テスト（282テスト）が通過
- [ ] カバレッジ > 90%
- [ ] パフォーマンス劣化なし

**PR**: `feature/81-phase5-tests` → `main`

---

## 全体の進捗

### マイルストーン
- [ ] **Phase 1 完了** - 不変性の回復（2-3日）
- [ ] **Phase 2 完了** - スキーマ標準化（2日）
- [ ] **Phase 3 完了** - WebUI 対応（2-3日）
- [ ] **Phase 4 完了** - CLI 統合（1日）
- [ ] **Phase 5 完了** - テスト拡充（2日）

### 最終完了条件
- [ ] 全 282 テスト通過
- [ ] レビュアーの指摘事項がすべて解決
- [ ] ドキュメント更新完了
- [ ] README に Agentic Loop の使い方を追記

---

## 関連ドキュメント
- 詳細設計: `issues/81_redesign_task_breakdown.md`
- サマリー: `issues/81_redesign_summary.md`
- 元の設計: `issues/81_agentic_loop_design.md`

---

## ノート（実装中のメモ）

### Phase 1 実装メモ
<!-- ここに Phase 1 実装中の気づきやメモを記載 -->

### Phase 2 実装メモ
<!-- ここに Phase 2 実装中の気づきやメモを記載 -->

### Phase 3 実装メモ
<!-- ここに Phase 3 実装中の気づきやメモを記載 -->
