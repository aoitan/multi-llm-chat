# Issue #81 再設計：タスク分割案

## 問題の分析

レビューで指摘された Critical Issues：

1. **アーキテクチャの境界侵犯**
   - `execute_with_tools()` が `history` を直接変更
   - 不変性の原則違反、テスト困難、競合リスク

2. **非同期への全面移行の破壊的影響**
   - 既存の同期コードとの互換性喪失
   - `asyncio.run()` のネスト実行でデッドロックリスク

3. **履歴ロールの一貫性崩壊**
   - `role: "tool"` を無秩序に導入
   - `history_utils.py` の検証ロジックと不整合

4. **MCPクライアントのライフサイクル管理の欠如**
   - WebUI でのマルチセッション対応が未設計

---

## 解決策：段階的実装アプローチ

### Phase 1: 基盤修正（Critical、ブロッキング）
**目的**: 既存機能を壊さずに、アーキテクチャの健全性を回復

#### Task 1.1: `execute_with_tools()` の不変性対応
**ファイル**: `src/multi_llm_chat/core.py`

**変更内容**:
```python
@dataclass
class AgenticLoopResult:
    """Agentic Loop の実行結果（不変オブジェクト）"""
    chunks: List[Dict[str, Any]]  # ストリーミングチャンク
    history_delta: List[Dict[str, Any]]  # 追加する履歴エントリ
    final_text: str  # 最終的なテキスト応答
    iterations_used: int
    timed_out: bool
    error: Optional[str] = None

async def execute_with_tools(
    provider,
    history: List[Dict],  # 読み取り専用（コピーを作成）
    system_prompt: Optional[str] = None,
    mcp_client = None,
    max_iterations: int = 10,
    timeout: float = 120.0,
) -> AgenticLoopResult:
    """
    履歴を変更せず、結果オブジェクトを返す。
    呼び出し元が履歴を更新する責任を持つ。
    """
    working_history = copy.deepcopy(history)
    chunks = []
    
    # ... ループ処理（working_history を変更）...
    
    return AgenticLoopResult(
        chunks=chunks,
        history_delta=working_history[len(history):],
        final_text=final_text,
        iterations_used=iteration,
        timed_out=False
    )
```

**影響**:
- `execute_with_tools()` の戻り値の型が変わる
- 既存のテストを更新する必要あり

**テスト**:
- `test_execute_with_tools_returns_immutable_result()`
- `test_history_not_mutated_on_error()`

---

#### Task 1.2: 同期ラッパーの追加
**ファイル**: `src/multi_llm_chat/core.py`

**変更内容**:
```python
def execute_with_tools_sync(
    provider,
    history: List[Dict],
    **kwargs
) -> AgenticLoopResult:
    """
    同期環境用のラッパー。
    既存のイベントループが存在する場合はエラー。
    """
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "execute_with_tools_sync() cannot be called from async context. "
            "Use execute_with_tools() directly instead."
        )
    except RuntimeError:
        pass  # No running loop - safe to proceed
    
    return asyncio.run(execute_with_tools(provider, history, **kwargs))
```

**影響**:
- 既存の同期コードとの互換性を維持
- CLI からの呼び出しが容易に

**テスト**:
- `test_sync_wrapper_works_in_sync_context()`
- `test_sync_wrapper_raises_in_async_context()`

---

### Phase 2: 履歴スキーマの標準化（Critical）
**目的**: `role: "tool"` を正式に標準化し、検証ロジックを統一

#### Task 2.1: `history_utils.py` の拡張
**ファイル**: `src/multi_llm_chat/history_utils.py`

**変更内容**:
```python
# 既存
LLM_ROLES = {"assistant"}
USER_ROLES = {"user"}

# 新規追加
TOOL_ROLES = {"tool"}
ALL_ROLES = LLM_ROLES | USER_ROLES | TOOL_ROLES

def validate_history_entry(entry: Dict) -> None:
    """
    履歴エントリの妥当性を検証。
    role: "tool" を含む新スキーマに対応。
    """
    role = entry.get("role")
    if role not in ALL_ROLES:
        raise ValueError(f"Invalid role: {role}. Must be one of {ALL_ROLES}")
    
    content = entry.get("content", [])
    if not isinstance(content, list):
        raise ValueError(f"Content must be a list, got {type(content)}")
    
    # role: "tool" の場合、content は tool_result のみ
    if role == "tool":
        for item in content:
            if item.get("type") != "tool_result":
                raise ValueError(
                    f"role='tool' can only contain tool_result items, got {item.get('type')}"
                )
```

**影響**:
- `content_to_text()` が `role: "tool"` を正しく処理する必要あり
- 既存のテストで `role: "tool"` のケースを追加

**テスト**:
- `test_validate_history_with_tool_role()`
- `test_content_to_text_handles_tool_results()`

---

#### Task 2.2: Provider の `format_history()` 更新
**ファイル**: `src/multi_llm_chat/llm_provider.py`

**変更内容**:
```python
# GeminiProvider.format_history()
def format_history(cls, history):
    for entry in history:
        role = entry["role"]
        if role == "tool":
            # Gemini は role: "function" を使用
            # tool_result を function_response に変換
            for item in content:
                if item["type"] == "tool_result":
                    parts.append(
                        glm.Part(
                            function_response=glm.FunctionResponse(
                                name=item["name"],
                                response={"content": item["content"]}
                            )
                        )
                    )

# ChatGPTProvider.format_history()
def format_history(cls, history):
    for entry in history:
        role = entry["role"]
        if role == "tool":
            # OpenAI は role: "tool" をそのまま使用
            for item in content:
                if item["type"] == "tool_result":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item["tool_call_id"],
                        "content": item["content"]
                    })
```

**影響**:
- Issue #79, #80 で実装した format_history() の拡張
- 既存のテストケースに `role: "tool"` を含む履歴を追加

**テスト**:
- `test_gemini_format_history_with_tool_results()`
- `test_chatgpt_format_history_with_tool_results()`

---

### Phase 3: MCP クライアント管理（WebUI対応）
**目的**: マルチセッション環境での MCP クライアントのライフサイクル管理

#### Task 3.1: MCP Manager の実装
**ファイル**: `src/multi_llm_chat/webui/mcp_manager.py`（新規）

**変更内容**:
```python
from typing import Dict, Optional
from ..mcp.client import MCPClient

class MCPManager:
    """
    WebUI 用の MCP クライアント管理。
    セッションごとにクライアントを保持し、適切にクリーンアップ。
    """
    _clients: Dict[str, MCPClient] = {}
    
    @classmethod
    async def get_or_create_client(
        cls,
        session_id: str,
        server_command: str,
        server_args: list
    ) -> MCPClient:
        """セッション ID に対応するクライアントを取得または作成"""
        if session_id not in cls._clients:
            client = MCPClient(server_command, server_args)
            async with client:  # 接続確認
                cls._clients[session_id] = client
        return cls._clients[session_id]
    
    @classmethod
    async def close_client(cls, session_id: str) -> None:
        """セッション終了時にクライアントをクローズ"""
        if session_id in cls._clients:
            client = cls._clients.pop(session_id)
            await client.__aexit__(None, None, None)
    
    @classmethod
    async def close_all(cls) -> None:
        """全クライアントをクローズ（アプリ終了時）"""
        for client in cls._clients.values():
            await client.__aexit__(None, None, None)
        cls._clients.clear()
```

**影響**:
- WebUI の handlers.py から呼び出す
- Gradio のセッション管理と統合

**テスト**:
- `test_mcp_manager_per_session_isolation()`
- `test_mcp_manager_cleanup_on_close()`

---

#### Task 3.2: WebUI との統合
**ファイル**: `src/multi_llm_chat/webui/handlers.py`

**変更内容**:
```python
from .mcp_manager import MCPManager

async def respond_with_tools(
    message: str,
    history: list,
    session_id: str,  # Gradio が提供
    chat_service: ChatService,
) -> str:
    """
    WebUI のレスポンス生成（ツール対応版）
    """
    # MCP クライアント取得
    mcp_client = await MCPManager.get_or_create_client(
        session_id,
        server_command="uvx",
        server_args=["mcp-server-time"]
    )
    
    # Agentic Loop 実行
    result = await execute_with_tools(
        provider=chat_service.provider,
        history=history,
        mcp_client=mcp_client
    )
    
    # 履歴を更新（呼び出し元の責任）
    history.extend(result.history_delta)
    
    return result.final_text
```

**影響**:
- Gradio の `session_id` を使用してセッション分離
- アプリ終了時に `MCPManager.close_all()` を呼び出す

**テスト**:
- `test_webui_multi_session_tool_execution()`

---

### Phase 4: CLI 統合の簡素化
**目的**: CLI での Agentic Loop 使用を簡単にする

#### Task 4.1: CLI での同期ラッパー使用
**ファイル**: `src/multi_llm_chat/cli.py`

**変更内容**:
```python
def _process_with_tools(service, user_message, mcp_client):
    """
    CLI 用の簡易ラッパー。
    同期環境で execute_with_tools_sync() を使用。
    """
    result = execute_with_tools_sync(
        provider=service.provider,
        history=service.logic_history,
        system_prompt=service.system_prompt,
        mcp_client=mcp_client
    )
    
    # 履歴を更新
    service.logic_history.extend(result.history_delta)
    
    # UI に表示
    for chunk in result.chunks:
        if chunk["type"] == "tool_call":
            _display_tool_response("tool_call", chunk["content"])
        elif chunk["type"] == "tool_result":
            _display_tool_response("tool_result", chunk["content"])
    
    print(result.final_text)
```

**影響**:
- 既存の CLI フローを維持
- MCP クライアントは main() で初期化して渡す

**テスト**:
- `test_cli_with_tools_sync_wrapper()`

---

### Phase 5: テストカバレッジの拡充
**目的**: 新スキーマと並行処理のテストを追加

#### Task 5.1: 新テストケースの追加

**ファイル**: `tests/test_agentic_loop.py`
- `test_result_immutability()` - 結果オブジェクトが不変であることを確認
- `test_history_not_mutated()` - 元の history が変更されないことを確認
- `test_role_tool_in_history()` - `role: "tool"` の履歴エントリを正しく処理
- `test_parallel_execution()` - 複数のループを並行実行しても競合しない

**ファイル**: `tests/test_history_utils.py`
- `test_validate_tool_role()` - `role: "tool"` の検証
- `test_content_to_text_with_tool_results()` - tool_result の文字列化

**ファイル**: `tests/test_webui.py`
- `test_mcp_manager_session_isolation()` - セッション間の分離
- `test_webui_multi_user_tool_execution()` - 複数ユーザーの並行実行

---

## タスクの依存関係と優先順位

```
Priority 1 (ブロッキング):
  Task 1.1 (不変性対応) ──┐
  Task 1.2 (同期ラッパー)  ├─→ Task 2.1 (スキーマ標準化)
                            └─→ Task 2.2 (format_history更新)

Priority 2 (WebUI対応):
  Task 3.1 (MCP Manager) ──→ Task 3.2 (WebUI統合)

Priority 3 (統合):
  Task 4.1 (CLI統合)
  Task 5.1 (テスト拡充)
```

---

## 実装スケジュール案

### Week 1: Phase 1 + Phase 2
- Day 1-2: Task 1.1 (不変性対応) + テスト
- Day 3: Task 1.2 (同期ラッパー) + テスト
- Day 4-5: Task 2.1, 2.2 (スキーマ標準化) + テスト

### Week 2: Phase 3 + Phase 4 + Phase 5
- Day 1-2: Task 3.1, 3.2 (MCP Manager) + テスト
- Day 3: Task 4.1 (CLI統合) + テスト
- Day 4-5: Task 5.1 (テストカバレッジ) + 統合テスト

---

## ロールバック計画

各 Phase が完了するまで、既存の実装（`feature/81-agentic-loop` ブランチ）は **マージしない**。

Phase 1 が完了した時点で、以下を確認：
1. 全既存テストが通過
2. 新しい `AgenticLoopResult` 型が正しく動作
3. パフォーマンス劣化なし（ベンチマーク）

問題があれば、Phase 1 を破棄して設計を再検討。

---

## 成功の定義

### Phase 1 完了の条件
- [ ] `execute_with_tools()` が history を変更しない
- [ ] `AgenticLoopResult` がすべての必要情報を含む
- [ ] 同期ラッパーが既存コードと互換性を持つ
- [ ] 全既存テスト + 新規テスト（10件）が通過

### Phase 2 完了の条件
- [ ] `role: "tool"` が `history_utils.py` で検証される
- [ ] `format_history()` が `role: "tool"` を正しく処理
- [ ] 全既存テスト + 新規テスト（5件）が通過

### Phase 3 完了の条件
- [ ] `MCPManager` がセッション分離を実現
- [ ] WebUI で複数ユーザーが同時にツールを使用可能
- [ ] リソースリークなし（メモリプロファイリング）

### 全Phase 完了の条件
- [ ] 全262テスト + 新規テスト（20件） = 282テスト通過
- [ ] レビュアーの指摘事項がすべて解決
- [ ] ドキュメント更新完了
