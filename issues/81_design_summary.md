# Issue #81 設計サマリー

## 主要な設計決定

### 1. Agentic Loop のアーキテクチャ
- **3レイヤー構成**: MCPClient → Core Logic → UI
- **非同期ジェネレーター**: ストリーミングレスポンスを維持
- **履歴の変異**: `execute_with_tools()`が`history`を直接更新
- **UI通知**: `yield`で`tool_call`/`tool_result`イベントを伝播

### 2. ツール実行フロー
```
User Input
  ↓
LLM Call (with tools)
  ↓
Tool Call? ─No→ Return Text
  ↓ Yes
Execute via MCP
  ↓
Append tool_result to history
  ↓
Loop (max 10 iterations, 120s timeout)
```

### 3. エラーハンドリング戦略
- **ツール実行失敗**: エラーメッセージをLLMにフィードバック（リカバリー機会）
- **接続エラー**: 即座に`ConnectionError`を送出
- **タイムアウト**: `TimeoutError`を送出（UI層で処理）

### 4. 履歴フォーマット
Issue #79, #80 で確立された構造化コンテンツ形式を使用：

```python
# Tool call (assistant message)
{
    "role": "assistant",
    "content": [
        {
            "type": "tool_call",
            "content": {
                "name": "get_weather",
                "arguments": {"location": "Tokyo"},
                "tool_call_id": "call_123"  # OpenAI only
            }
        }
    ]
}

# Tool result (user message)
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "content": "25°C",
            "tool_call_id": "call_123",  # OpenAI only
            "name": "get_weather"
        }
    ]
}
```

---

## 実装の4フェーズ

### Phase 1: MCPClient 拡張
**ファイル:** `src/multi_llm_chat/mcp/client.py`

```python
async def call_tool(self, name: str, arguments: dict) -> dict:
    """Execute tool and return result."""
    response = await self.session.call_tool(name, arguments)
    return {
        "content": [{"type": item.type, ...} for item in response.content],
        "isError": response.isError,
    }
```

**テスト:** `tests/test_mcp_client.py` (2件)
- `test_call_tool_success`
- `test_call_tool_error`

---

### Phase 2: Core Logic 実装
**ファイル:** `src/multi_llm_chat/core.py`

```python
async def execute_with_tools(
    provider: LLMProvider,
    history: List[Dict],
    system_prompt: Optional[str] = None,
    mcp_client: Optional[MCPClient] = None,
    max_iterations: int = 10,
    timeout: float = 120.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Agentic Loop implementation."""
    tools = await mcp_client.list_tools()
    
    for iteration in range(max_iterations):
        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError(...)
        
        # Call LLM
        tool_calls_in_turn = []
        async for chunk in provider.call_api(history, system_prompt, tools):
            if chunk["type"] == "tool_call":
                tool_calls_in_turn.append(chunk["content"])
                yield chunk
            elif chunk["type"] == "text":
                yield chunk
        
        # No tool calls → final response
        if not tool_calls_in_turn:
            break
        
        # Execute tools
        for tool_call in tool_calls_in_turn:
            result = await mcp_client.call_tool(...)
            yield {"type": "tool_result", "content": ...}
        
        # Update history (structured content)
        history.append({"role": "assistant", "content": [...]})
        history.append({"role": "user", "content": [...]})
```

**テスト:** `tests/test_agentic_loop.py` (4件・新規)
- `test_execute_with_tools_single_iteration` (正常系)
- `test_execute_with_tools_max_iterations` (境界値)
- `test_execute_with_tools_timeout` (異常系)
- `test_execute_with_tools_tool_error` (異常系)

---

### Phase 3: CLI 統合
**ファイル:** `src/multi_llm_chat/cli.py`

```python
async def _handle_chat_response_with_tools(self, response_stream, mcp_client):
    """Display tool calls and results."""
    async for chunk in response_stream:
        if chunk["type"] == "tool_call":
            print(f"\n[Tool Call: {chunk['content']['name']}]")
        elif chunk["type"] == "tool_result":
            print(f"[Tool Result: {chunk['content']['name']}]")
            print(f"  {chunk['content']['content']}")
        elif chunk["type"] == "text":
            print(chunk["content"], end="", flush=True)
```

**テスト:** `tests/test_cli.py` (1件追加)
- `test_cli_with_tools`

---

### Phase 4: Web UI 統合
**ファイル:** `src/multi_llm_chat/webui/handlers.py`

```python
async def respond_with_tools(message, history, chat_service, mcp_client):
    """Gradio handler with Agentic Loop."""
    response_text = ""
    tool_calls_text = ""
    
    async for chunk in execute_with_tools(...):
        if chunk["type"] == "text":
            response_text += chunk["content"]
        elif chunk["type"] == "tool_call":
            tool_calls_text += f"\n\n🔧 **Tool Call**: {chunk['content']['name']}\n"
        elif chunk["type"] == "tool_result":
            tool_calls_text += f"✅ **Result**: {chunk['content']['content'][:100]}...\n"
        
        yield response_text + tool_calls_text
```

**テスト:** `tests/test_webui_handlers.py` (1件追加)
- `test_webui_respond_with_tools`

---

## テスト戦略まとめ

| Phase | テストファイル | テスト数 | 種類 |
|-------|----------------|----------|------|
| 1 | `test_mcp_client.py` | 2 | 単体（成功/失敗） |
| 2 | `test_agentic_loop.py` | 4 | 単体（正常/境界/異常×2） |
| 3 | `test_cli.py` | 1 | 統合（表示確認） |
| 4 | `test_webui_handlers.py` | 1 | 統合（表示確認） |
| **合計** | - | **8** | - |

**既存テスト:** 252件（Issue #80完了時点）  
**新規テスト:** 8件  
**合計:** 260件

---

## 重要な技術的決定

### 1. ツール結果の簡略化
**決定:** MCP の複雑な `CallToolResult`（text/image/resource）をテキストのみに簡略化してLLMにフィードバック。

**理由:**
- 現在のLLM APIは画像/リソースをtool_resultとしてサポートしていない
- 実装の単純化
- 将来のマルチモーダル対応時に拡張可能

```python
# MCP response (複雑)
{"content": [
    {"type": "text", "text": "Result"},
    {"type": "image", "data": "base64...", "mimeType": "image/png"}
]}

# LLM feedback (簡略化)
{"type": "tool_result", "content": "Result"}  # Text only
```

---

### 2. 非同期実装
**決定:** `execute_with_tools()`を非同期ジェネレーターとして実装。

**理由:**
- MCPClientが非同期API
- ストリーミングレスポンスの維持（リアルタイムUI更新）
- Gradioも非同期対応

**影響:**
- CLI/Web UIの統合コードも非同期化
- 既存の同期コードは`asyncio.run()`でラップ

---

### 3. ループ制御パラメータ

| パラメータ | デフォルト | 理由 |
|------------|------------|------|
| `max_iterations` | 10 | OpenAI Assistants APIの推奨値 |
| `timeout` | 120秒 | 複雑なツール呼び出しチェーンを考慮 |

**根拠:**
- OpenAI公式ドキュメント: https://platform.openai.com/docs/assistants/tools/function-calling
- LangChain AgentExecutorのデフォルト: 15イテレーション

---

### 4. エラーリカバリー戦略
**決定:** ツール実行失敗時、LLMにエラー内容を伝達してリカバリーを試みる。

```python
try:
    result = await mcp_client.call_tool(name, arguments)
except Exception as e:
    # Don't raise - let LLM handle the error
    tool_result = {"content": f"Tool execution failed: {e}"}
    yield {"type": "tool_result", "content": tool_result}
```

**メリット:**
- LLMが代替手段を提案できる
- ユーザーに対してより親切なエラーメッセージ
- 一部のツール失敗が全体のフローを止めない

**例外:** `ConnectionError`（MCPサーバーダウン）は即座に`raise`

---

## TDD実装順序

### Step 1-2: MCPClient (RED → GREEN)
1. **RED**: `tests/test_mcp_client.py`に2テスト追加 → `AttributeError`
2. **GREEN**: `src/multi_llm_chat/mcp/client.py`に`call_tool()`実装 → ✅

### Step 3-4: Core Logic (RED → GREEN)
3. **RED**: `tests/test_agentic_loop.py`を新規作成 → `ImportError`
4. **GREEN**: `src/multi_llm_chat/core.py`に`execute_with_tools()`実装 → ✅

### Step 5-6: CLI Integration (RED → GREEN)
5. **RED**: `tests/test_cli.py`に1テスト追加 → `AttributeError`
6. **GREEN**: `src/multi_llm_chat/cli.py`を拡張 → ✅

### Step 7-8: Web UI Integration (RED → GREEN)
7. **RED**: `tests/test_webui_handlers.py`に1テスト追加 → 失敗
8. **GREEN**: `src/multi_llm_chat/webui/handlers.py`を拡張 → ✅

---

## 制限事項

### 現時点の制限
1. **マルチモーダル tool_result 非対応** - 画像/リソースは未サポート
2. **並列ツール実行なし** - 順次実行のみ（実装の単純化）
3. **ツール選択の制御なし** - `tool_choice="auto"`固定
4. **ストリーミング中断不可** - Ctrl+Cまたは強制終了のみ

### 将来的な拡張候補
- 並列ツール実行（`asyncio.gather()`）
- `tool_choice`パラメータの公開
- ストリーミング中断機能
- マルチモーダル tool_result 対応

---

## 受け入れ条件チェックリスト

### 機能要件
- [ ] `MCPClient.call_tool()`が実装され、ツールを実行できる
- [ ] `execute_with_tools()`がAgentic Loopを実装
- [ ] CLIでツール呼び出し/結果が表示される（`[Tool Call: ...]`、`[Tool Result: ...]`）
- [ ] Web UIでツール呼び出し/結果がMarkdown形式で表示される（🔧, ✅）
- [ ] max_iterations到達時に警告ログを出力
- [ ] タイムアウト超過時に`TimeoutError`を送出
- [ ] ツール実行失敗時にエラーをLLMにフィードバック

### テスト要件
- [ ] `tests/test_mcp_client.py`に2テスト追加
- [ ] `tests/test_agentic_loop.py`を新規作成（4テスト）
- [ ] `tests/test_cli.py`に1テスト追加
- [ ] `tests/test_webui_handlers.py`に1テスト追加
- [ ] 全260テストが通過
- [ ] Ruff lint/formatが通過

### ドキュメント要件
- [ ] `doc/agentic_loop_guide.md`を作成
- [ ] READMEに機能追記

---

## 次のステップ（Issue #81完了後）

1. **Issue #82**: Web UI でのMCPサーバー設定UI
2. **Issue #83**: Filesystem MCPサーバー統合
3. **Story #78完了**: 全体的な統合テストとドキュメント整備
