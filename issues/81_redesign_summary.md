# Issue #81 再設計サマリー

## 🚨 現状の問題

レビュアーが指摘した Critical Issues：

1. **アーキテクチャ違反**: `history` の直接変更（5箇所）→ 不変性原則違反
2. **非同期強制**: 同期ラッパーなし → 既存コードとの互換性喪失
3. **スキーマの無秩序**: `role: "tool"` を検証なしで導入
4. **ライフサイクル欠如**: MCP クライアントのマルチセッション対応なし

---

## ✅ 解決アプローチ

### 最優先タスク（Phase 1）: 不変性の回復

#### Task 1.1: `AgenticLoopResult` の導入

**Before (問題のあるコード)**:
```python
async def execute_with_tools(provider, history, ...):
    history.append(...)  # ❌ 副作用
    yield chunk
```

**After (修正後)**:
```python
@dataclass
class AgenticLoopResult:
    chunks: List[Dict[str, Any]]
    history_delta: List[Dict[str, Any]]  # 追加分のみ
    final_text: str
    iterations_used: int
    timed_out: bool

async def execute_with_tools(
    provider,
    history: List[Dict],  # 読み取り専用
    ...
) -> AgenticLoopResult:
    working_copy = copy.deepcopy(history)
    # working_copy を変更
    return AgenticLoopResult(
        history_delta=working_copy[len(history):]
    )
```

**呼び出し側の責任**:
```python
result = await execute_with_tools(provider, history, ...)
history.extend(result.history_delta)  # 明示的に更新
```

---

#### Task 1.2: 同期ラッパーの追加

```python
def execute_with_tools_sync(...) -> AgenticLoopResult:
    """同期環境用のラッパー"""
    try:
        asyncio.get_running_loop()
        raise RuntimeError("Cannot call from async context")
    except RuntimeError:
        pass
    return asyncio.run(execute_with_tools(...))
```

---

### Phase 2: スキーマ標準化

#### Task 2.1: `history_utils.py` の拡張

```python
TOOL_ROLES = {"tool"}
ALL_ROLES = LLM_ROLES | USER_ROLES | TOOL_ROLES

def validate_history_entry(entry):
    if entry["role"] not in ALL_ROLES:
        raise ValueError(f"Invalid role: {entry['role']}")
    
    if entry["role"] == "tool":
        for item in entry["content"]:
            if item["type"] != "tool_result":
                raise ValueError("role='tool' can only contain tool_result")
```

---

### Phase 3: MCP Manager（WebUI対応）

#### Task 3.1: セッション分離

```python
class MCPManager:
    _clients: Dict[str, MCPClient] = {}
    
    @classmethod
    async def get_or_create_client(cls, session_id, ...):
        if session_id not in cls._clients:
            cls._clients[session_id] = MCPClient(...)
        return cls._clients[session_id]
```

---

## 📊 タスク依存関係

```
[Task 1.1: 不変性] ─┐
[Task 1.2: 同期]    ├→ [Task 2.1: スキーマ] → [Task 2.2: format_history]
                    └→ [Task 3.1: MCP Manager] → [Task 3.2: WebUI統合]
                       └→ [Task 4.1: CLI統合]
```

---

## 🎯 実装優先順位

### 今すぐ着手すべき (Priority 1)
1. **Task 1.1**: `AgenticLoopResult` + 不変性対応
2. **Task 1.2**: 同期ラッパー

### 次に着手 (Priority 2)
3. **Task 2.1**: `TOOL_ROLES` 定義 + 検証ロジック
4. **Task 2.2**: `format_history()` の `role: "tool"` 対応

### 最後に統合 (Priority 3)
5. **Task 3.1**: `MCPManager` 実装
6. **Task 3.2**: WebUI 統合
7. **Task 4.1**: CLI 統合
8. **Task 5.1**: テストカバレッジ拡充

---

## 📝 成功の指標

### Phase 1 完了（ブロッキング解除）
- [ ] `execute_with_tools()` が history を変更しない
- [ ] 同期ラッパーで既存コードと互換性
- [ ] 全既存テスト + 新規10テスト 通過

### Phase 2 完了（標準化）
- [ ] `role: "tool"` が正式に検証される
- [ ] Gemini/ChatGPT 両方で正しく動作
- [ ] 全テスト + 新規5テスト 通過

### Phase 3 完了（WebUI対応）
- [ ] マルチセッション分離動作
- [ ] リソースリークなし
- [ ] 並行実行テスト通過

---

## ⚠️ ロールバック条件

Phase 1 で以下の問題が発生した場合、設計を再検討：
- パフォーマンス劣化 > 10%
- 既存テストの修正コスト > 2日
- `deepcopy` がボトルネックになる

---

## 📅 推奨スケジュール

- **Week 1, Day 1-3**: Phase 1 (Task 1.1, 1.2)
- **Week 1, Day 4-5**: Phase 2 (Task 2.1, 2.2)
- **Week 2, Day 1-3**: Phase 3 (Task 3.1, 3.2)
- **Week 2, Day 4-5**: Phase 4 + Phase 5 (統合 + テスト)

---

## 🔧 次のアクション

1. **ブランチをリセット**: `feature/81-agentic-loop` を破棄し、`feature/81-phase1-immutability` を作成
2. **Task 1.1 実装開始**: `AgenticLoopResult` クラスの定義から始める
3. **レビュー**: Phase 1 完了後、再度レビュアーに確認

---

## 質問事項（ユーザー確認）

1. **Phase 1 を最優先で着手**してよいですか？
2. **既存の `feature/81-agentic-loop` ブランチを破棄**してよいですか？
3. **Phase ごとに PR を分割**する方針で進めてよいですか？
