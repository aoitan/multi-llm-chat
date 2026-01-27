# Issue #81: ツール実行ループの実装（Agentic Loop）

## 概要

LLMがツール呼び出しを行った際に、実際にMCPサーバーのツールを実行し、結果をLLMにフィードバックする「Agentic Loop」を実装する。これにより、LLMが複数回のツール呼び出しを繰り返して目標を達成できるようになる。

## 親Story
- #78: LLMツール実行統合

## 先行タスク（完了済み）
- ✅ Issue #79: GeminiProviderのtools対応（PR #94）
  - MCP → Gemini形式変換
  - ストリーミングツール呼び出し処理
  - `format_history()`でtool_call/tool_result対応
- ✅ Issue #80: ChatGPTProviderのtools対応（PR #97）
  - MCP → OpenAI形式変換
  - ストリーミングツール呼び出し処理
  - `format_history()`でtool_call/tool_result対応

## 現状分析

### 実装済み機能
1. **MCPクライアント**（`src/multi_llm_chat/mcp/client.py`）
   - ✅ `list_tools()`: ツール定義の取得
   - ❌ `call_tool()`: ツール実行（未実装）

2. **Provider層**（`src/multi_llm_chat/llm_provider.py`）
   - ✅ `call_api(tools=...)`でツール定義を渡せる
   - ✅ ストリーミングレスポンスで`{"type": "tool_call", "content": {...}}`を返す
   - ✅ `format_history()`でtool_call/tool_resultを含む履歴をフォーマット

3. **チャットロジック**（`src/multi_llm_chat/chat_logic.py`）
   - ✅ ツール呼び出しイベントを受信して履歴に追加（187行目）
   - ❌ ツール実行 + 結果フィードバックループ（未実装）

### 不足している機能
1. **MCPClientのツール実行**: `call_tool(name, arguments)` メソッド
2. **Agentic Loopのコアロジック**: ツール呼び出し → 実行 → 結果フィードバック → 再呼び出し
3. **ループ制御**: 最大反復回数、タイムアウト、エラーハンドリング
4. **CLI/Web UI統合**: 両方のインターフェースで動作する設計

---

## 設計方針

### 1. アーキテクチャ概要

```
┌─────────────────┐
│  CLI / Web UI   │
└────────┬────────┘
         │ ユーザー入力
         ↓
┌─────────────────────────────────────────────────────┐
│  Agentic Loop (core.py)                             │
│  ┌──────────────────────────────────────┐           │
│  │  Loop until:                         │           │
│  │  - LLM returns text (no tool calls)  │           │
│  │  - Max iterations reached            │           │
│  │  - Error occurred                    │           │
│  └──────────────────────────────────────┘           │
│         │                                            │
│         ↓                                            │
│  ┌──────────────────────────────────────┐           │
│  │  1. Call LLM with history + tools    │           │
│  └──────────────────────────────────────┘           │
│         │                                            │
│         ↓                                            │
│  ┌──────────────────────────────────────┐           │
│  │  2. If tool_call:                    │           │
│  │     - Execute via MCPClient          │           │
│  │     - Append tool_result to history  │           │
│  │     - Continue loop                  │           │
│  └──────────────────────────────────────┘           │
│         │                                            │
│         ↓                                            │
│  ┌──────────────────────────────────────┐           │
│  │  3. If text:                         │           │
│  │     - Return final response          │           │
│  └──────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### 2. 実装レイヤー

#### Phase 1: MCPClientの拡張
**ファイル:** `src/multi_llm_chat/mcp/client.py`

```python
async def call_tool(self, name: str, arguments: dict) -> dict:
    """Execute a tool on the MCP server.
    
    Args:
        name: Tool name (e.g., "get_weather")
        arguments: Tool arguments as dict (e.g., {"location": "Tokyo"})
    
    Returns:
        Tool result with structure:
        {
            "content": [
                {"type": "text", "text": "..."},
                # or {"type": "image", "data": "...", "mimeType": "..."},
                # or {"type": "resource", "resource": {...}}
            ],
            "isError": bool  # Optional, indicates execution failure
        }
    
    Raises:
        ConnectionError: If session is not initialized
        ValueError: If tool execution fails
    """
    if not self.session:
        raise ConnectionError("Client is not connected.")
    
    response = await self.session.call_tool(name, arguments)
    return {
        "content": [
            {"type": item.type, **item.model_dump(exclude={"type"})}
            for item in response.content
        ],
        "isError": response.isError if hasattr(response, "isError") else False,
    }
```

**設計ポイント:**
- MCP仕様の`CallToolResult`をそのまま返す（上位層で加工）
- `isError`フィールドでツール実行失敗を判定可能
- `content`は複数の型（text/image/resource）を持ちうる

---

#### Phase 2: Agentic Loopのコアロジック
**ファイル:** `src/multi_llm_chat/core.py`（新規関数）

```python
async def execute_with_tools(
    provider: LLMProvider,
    history: List[Dict],
    system_prompt: Optional[str] = None,
    mcp_client: Optional[MCPClient] = None,
    max_iterations: int = 10,
    timeout: float = 120.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute LLM call with Agentic Loop for tool execution.
    
    Repeatedly calls LLM and executes tools until:
    - LLM returns text without tool calls
    - max_iterations is reached
    - timeout is exceeded
    - An error occurs
    
    Args:
        provider: LLM provider instance (Gemini/ChatGPT)
        history: Conversation history (will be mutated)
        system_prompt: Optional system prompt
        mcp_client: Optional MCP client for tool execution
        max_iterations: Maximum number of LLM calls (default: 10)
        timeout: Maximum total execution time in seconds (default: 120)
    
    Yields:
        Streaming chunks from LLM:
        - {"type": "text", "content": str}
        - {"type": "tool_call", "content": {...}}
        - {"type": "tool_result", "content": {...}}
    
    Raises:
        TimeoutError: If execution exceeds timeout
        ValueError: If tool execution fails critically
    """
    import time
    
    start_time = time.time()
    tools = await mcp_client.list_tools() if mcp_client else None
    
    for iteration in range(max_iterations):
        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Agentic loop exceeded {timeout}s timeout")
        
        # Call LLM
        tool_calls_in_turn = []
        async for chunk in provider.call_api(history, system_prompt, tools):
            chunk_type = chunk.get("type")
            
            if chunk_type == "text":
                yield chunk
            elif chunk_type == "tool_call":
                tool_call = chunk.get("content", {})
                tool_calls_in_turn.append(tool_call)
                yield chunk  # Notify UI
        
        # If no tool calls, loop ends (final response)
        if not tool_calls_in_turn:
            break
        
        # Execute tools and collect results
        tool_results = []
        for tool_call in tool_calls_in_turn:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            tool_call_id = tool_call.get("tool_call_id")  # OpenAI only
            
            try:
                result = await mcp_client.call_tool(name, arguments)
                
                # Extract text content (simplify for LLM)
                text_parts = [
                    item.get("text", "")
                    for item in result.get("content", [])
                    if item.get("type") == "text"
                ]
                result_text = "\n".join(text_parts) if text_parts else "(no text output)"
                
                tool_result = {
                    "name": name,
                    "content": result_text,
                }
                if tool_call_id:
                    tool_result["tool_call_id"] = tool_call_id
                
                tool_results.append(tool_result)
                
                # Notify UI
                yield {"type": "tool_result", "content": tool_result}
                
            except Exception as e:
                # Tool execution failed - report to LLM
                error_text = f"Tool execution failed: {str(e)}"
                tool_result = {
                    "name": name,
                    "content": error_text,
                }
                if tool_call_id:
                    tool_result["tool_call_id"] = tool_call_id
                
                tool_results.append(tool_result)
                yield {"type": "tool_result", "content": tool_result}
        
        # Append tool_call and tool_result to history
        # Note: History format is structured content (list of items)
        assistant_entry = {"role": "assistant", "content": []}
        for tc in tool_calls_in_turn:
            assistant_entry["content"].append({"type": "tool_call", "content": tc})
        history.append(assistant_entry)
        
        # Tool results as separate user message
        user_entry = {"role": "user", "content": []}
        for tr in tool_results:
            tool_result_item = {"type": "tool_result", "content": tr["content"]}
            if "tool_call_id" in tr:
                tool_result_item["tool_call_id"] = tr["tool_call_id"]
            if "name" in tr:
                tool_result_item["name"] = tr["name"]
            user_entry["content"].append(tool_result_item)
        history.append(user_entry)
    
    # Max iterations reached
    if iteration == max_iterations - 1:
        logger.warning("Agentic loop reached max_iterations=%d", max_iterations)
```

**設計ポイント:**
- **非同期ジェネレーター**: ストリーミングレスポンスを維持
- **履歴の変異**: `history`を直接更新（呼び出し元で永続化）
- **タイムアウト制御**: `time.time()`で経過時間を監視
- **エラーハンドリング**: ツール実行失敗を`tool_result`としてLLMに伝達
- **UI通知**: `{"type": "tool_result", ...}`を`yield`してUI更新

---

#### Phase 3: CLI統合
**ファイル:** `src/multi_llm_chat/cli.py`

既存の`_handle_chat_response()`を拡張してツール実行をサポート。

```python
async def _handle_chat_response_with_tools(
    self,
    response_stream,
    mcp_client: Optional[MCPClient] = None
):
    """Handle streaming response with Agentic Loop.
    
    Displays tool calls and tool results in real-time.
    """
    content_parts = []
    
    async for chunk in response_stream:
        chunk_type = chunk.get("type")
        
        if chunk_type == "text":
            text = chunk.get("content", "")
            print(text, end="", flush=True)
            content_parts.append({"type": "text", "content": text})
        
        elif chunk_type == "tool_call":
            tool_call = chunk.get("content", {})
            name = tool_call.get("name")
            args = tool_call.get("arguments", {})
            
            # Display tool call
            print(f"\n[Tool Call: {name}]", flush=True)
            print(f"  Args: {args}", flush=True)
            
            content_parts.append({"type": "tool_call", "content": tool_call})
        
        elif chunk_type == "tool_result":
            result = chunk.get("content", {})
            name = result.get("name")
            content = result.get("content")
            
            # Display tool result
            print(f"[Tool Result: {name}]", flush=True)
            print(f"  {content}", flush=True)
            print()  # Blank line
    
    print()  # Final newline
    return content_parts
```

**変更点:**
- `respond()`メソッドで`execute_with_tools()`を呼び出し
- ツール実行結果を視覚的に表示（`[Tool Call: ...]`、`[Tool Result: ...]`）

---

#### Phase 4: Web UI統合
**ファイル:** `src/multi_llm_chat/webui/handlers.py`

Gradioの`respond()`関数でAgentic Loopを統合。

```python
async def respond_with_tools(
    message,
    history,
    chat_service,
    mcp_client=None
):
    """Web UI handler with Agentic Loop support."""
    # Convert Gradio history to internal format
    internal_history = convert_gradio_to_internal(history)
    
    # Add user message
    internal_history.append({"role": "user", "content": message})
    
    # Stream response with Agentic Loop
    response_text = ""
    tool_calls_text = ""
    
    async for chunk in execute_with_tools(
        chat_service.provider,
        internal_history,
        chat_service.system_prompt,
        mcp_client
    ):
        chunk_type = chunk.get("type")
        
        if chunk_type == "text":
            response_text += chunk.get("content", "")
            yield response_text + tool_calls_text
        
        elif chunk_type == "tool_call":
            tool_call = chunk.get("content", {})
            tool_calls_text += f"\n\n🔧 **Tool Call**: {tool_call['name']}\n"
            yield response_text + tool_calls_text
        
        elif chunk_type == "tool_result":
            result = chunk.get("content", {})
            tool_calls_text += f"✅ **Result**: {result['content'][:100]}...\n"
            yield response_text + tool_calls_text
    
    # Update history
    history.append([message, response_text])
```

**設計ポイント:**
- ツール呼び出し/結果をMarkdown形式で表示（🔧, ✅アイコン）
- Gradioのストリーミング更新を維持
- 最終的な`response_text`のみを履歴に保存（ツール情報は視覚効果のみ）

---

## テスト戦略

### Phase 1: MCPClient.call_tool()のテスト
**ファイル:** `tests/test_mcp_client.py`（既存）

```python
@pytest.mark.asyncio
async def test_call_tool_success():
    """Test successful tool execution."""
    mock_session = AsyncMock()
    mock_session.call_tool.return_value = MockToolResult(
        content=[MockTextContent(type="text", text="Tokyo weather: 25°C")],
        isError=False,
    )
    
    client = MCPClient("uvx", ["mcp-server-weather"])
    client.session = mock_session
    
    result = await client.call_tool("get_weather", {"location": "Tokyo"})
    
    assert result["content"][0]["type"] == "text"
    assert "25°C" in result["content"][0]["text"]
    assert result["isError"] is False

@pytest.mark.asyncio
async def test_call_tool_error():
    """Test tool execution failure."""
    mock_session = AsyncMock()
    mock_session.call_tool.return_value = MockToolResult(
        content=[MockTextContent(type="text", text="API key missing")],
        isError=True,
    )
    
    client = MCPClient("uvx", ["mcp-server-weather"])
    client.session = mock_session
    
    result = await client.call_tool("get_weather", {"location": "Invalid"})
    
    assert result["isError"] is True
    assert "API key missing" in result["content"][0]["text"]
```

### Phase 2: Agentic Loop単体テスト
**ファイル:** `tests/test_agentic_loop.py`（新規）

```python
@pytest.mark.asyncio
async def test_execute_with_tools_single_iteration():
    """Test single tool call and final response."""
    mock_provider = AsyncMock()
    mock_provider.call_api.side_effect = [
        # First call: tool_call
        async_generator([
            {"type": "tool_call", "content": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}
        ]),
        # Second call: final text
        async_generator([
            {"type": "text", "content": "The weather in Tokyo is 25°C."}
        ]),
    ]
    
    mock_mcp = AsyncMock()
    mock_mcp.list_tools.return_value = [{"name": "get_weather", ...}]
    mock_mcp.call_tool.return_value = {
        "content": [{"type": "text", "text": "25°C"}],
        "isError": False,
    }
    
    history = []
    chunks = []
    async for chunk in execute_with_tools(mock_provider, history, mcp_client=mock_mcp):
        chunks.append(chunk)
    
    # Verify: tool_call → tool_result → text
    assert chunks[0]["type"] == "tool_call"
    assert chunks[1]["type"] == "tool_result"
    assert chunks[2]["type"] == "text"
    
    # Verify history structure
    assert len(history) == 2  # assistant (tool_call) + user (tool_result)
    assert history[0]["role"] == "assistant"
    assert history[0]["content"][0]["type"] == "tool_call"
    assert history[1]["role"] == "user"
    assert history[1]["content"][0]["type"] == "tool_result"

@pytest.mark.asyncio
async def test_execute_with_tools_max_iterations():
    """Test loop termination at max_iterations."""
    mock_provider = AsyncMock()
    mock_provider.call_api.return_value = async_generator([
        {"type": "tool_call", "content": {"name": "infinite_tool", "arguments": {}}}
    ])
    
    mock_mcp = AsyncMock()
    mock_mcp.list_tools.return_value = [{"name": "infinite_tool", ...}]
    mock_mcp.call_tool.return_value = {"content": [{"type": "text", "text": "done"}], "isError": False}
    
    history = []
    chunks = []
    async for chunk in execute_with_tools(mock_provider, history, mcp_client=mock_mcp, max_iterations=3):
        chunks.append(chunk)
    
    # Should stop at 3 iterations
    assert len([c for c in chunks if c["type"] == "tool_call"]) == 3

@pytest.mark.asyncio
async def test_execute_with_tools_timeout():
    """Test timeout handling."""
    mock_provider = AsyncMock()
    
    async def slow_generator():
        await asyncio.sleep(2)  # Simulate slow response
        yield {"type": "text", "content": "done"}
    
    mock_provider.call_api.return_value = slow_generator()
    
    with pytest.raises(TimeoutError):
        async for _ in execute_with_tools(mock_provider, [], timeout=0.5):
            pass

@pytest.mark.asyncio
async def test_execute_with_tools_tool_error():
    """Test graceful handling of tool execution errors."""
    mock_provider = AsyncMock()
    mock_provider.call_api.side_effect = [
        async_generator([
            {"type": "tool_call", "content": {"name": "failing_tool", "arguments": {}}}
        ]),
        async_generator([
            {"type": "text", "content": "I encountered an error."}
        ]),
    ]
    
    mock_mcp = AsyncMock()
    mock_mcp.list_tools.return_value = [{"name": "failing_tool", ...}]
    mock_mcp.call_tool.side_effect = Exception("Tool crashed")
    
    history = []
    chunks = []
    async for chunk in execute_with_tools(mock_provider, history, mcp_client=mock_mcp):
        chunks.append(chunk)
    
    # Should yield tool_result with error message
    tool_result = next(c for c in chunks if c["type"] == "tool_result")
    assert "Tool execution failed" in tool_result["content"]["content"]
```

### Phase 3: CLI統合テスト
**ファイル:** `tests/test_cli.py`（既存に追加）

```python
@pytest.mark.asyncio
async def test_cli_with_tools(capsys):
    """Test CLI displays tool calls and results."""
    mock_response = async_generator([
        {"type": "tool_call", "content": {"name": "get_weather", "arguments": {"location": "Tokyo"}}},
        {"type": "tool_result", "content": {"name": "get_weather", "content": "25°C"}},
        {"type": "text", "content": "The weather is 25°C."},
    ])
    
    cli = ChatCLI()
    content = await cli._handle_chat_response_with_tools(mock_response)
    
    captured = capsys.readouterr()
    assert "[Tool Call: get_weather]" in captured.out
    assert "[Tool Result: get_weather]" in captured.out
    assert "The weather is 25°C." in captured.out
```

### Phase 4: Web UI統合テスト
**ファイル:** `tests/test_webui_handlers.py`（既存に追加）

```python
@pytest.mark.asyncio
async def test_webui_respond_with_tools():
    """Test Web UI streaming with Agentic Loop."""
    mock_chat_service = Mock()
    mock_provider = AsyncMock()
    mock_provider.call_api.side_effect = [
        async_generator([
            {"type": "tool_call", "content": {"name": "search", "arguments": {}}},
        ]),
        async_generator([
            {"type": "text", "content": "Found results."},
        ]),
    ]
    
    mock_mcp = AsyncMock()
    mock_mcp.list_tools.return_value = [{"name": "search", ...}]
    mock_mcp.call_tool.return_value = {"content": [{"type": "text", "text": "data"}], "isError": False}
    
    mock_chat_service.provider = mock_provider
    
    responses = []
    async for response in respond_with_tools("search weather", [], mock_chat_service, mock_mcp):
        responses.append(response)
    
    # Should contain tool call indicator
    assert any("🔧 **Tool Call**" in r for r in responses)
    assert any("✅ **Result**" in r for r in responses)
    assert responses[-1] == "Found results."
```

---

## 実装順序（TDDサイクル）

### Step 1: RED - MCPClient.call_tool()のテスト作成
- `tests/test_mcp_client.py`に2つのテストケース追加
- pytest実行 → `AttributeError: 'MCPClient' object has no attribute 'call_tool'`

### Step 2: GREEN - MCPClient.call_tool()の実装
- `src/multi_llm_chat/mcp/client.py`に`call_tool()`メソッド追加
- pytest実行 → 全テスト通過

### Step 3: RED - Agentic Loopのテスト作成
- `tests/test_agentic_loop.py`を新規作成（4テストケース）
- pytest実行 → `ImportError: cannot import name 'execute_with_tools'`

### Step 4: GREEN - execute_with_tools()の実装
- `src/multi_llm_chat/core.py`に`execute_with_tools()`関数追加
- pytest実行 → 全テスト通過

### Step 5: RED - CLI統合テスト作成
- `tests/test_cli.py`に1テストケース追加
- pytest実行 → `AttributeError: 'ChatCLI' object has no attribute '_handle_chat_response_with_tools'`

### Step 6: GREEN - CLI統合実装
- `src/multi_llm_chat/cli.py`を拡張
- pytest実行 → 全テスト通過

### Step 7: RED - Web UI統合テスト作成
- `tests/test_webui_handlers.py`に1テストケース追加
- pytest実行 → 失敗

### Step 8: GREEN - Web UI統合実装
- `src/multi_llm_chat/webui/handlers.py`を拡張
- pytest実行 → 全テスト通過

---

## 設計上の重要な決定事項

### 1. 履歴フォーマット
Issue #79, #80で確立された構造化コンテンツ形式を踏襲：

```python
{
    "role": "assistant",
    "content": [
        {"type": "tool_call", "content": {"name": str, "arguments": dict, "tool_call_id": str}}
    ]
}
{
    "role": "user",
    "content": [
        {"type": "tool_result", "content": str, "tool_call_id": str, "name": str}
    ]
}
```

**理由:**
- Issue #80で実装した`format_history()`がこの形式をサポート済み
- Gemini/ChatGPT両方で統一的に処理可能

### 2. ツール結果の簡略化
MCP仕様の`CallToolResult`は複雑（text/image/resource）だが、LLMへのフィードバックはテキストのみに簡略化。

```python
# MCP response (complex)
{
    "content": [
        {"type": "text", "text": "Result 1"},
        {"type": "image", "data": "base64...", "mimeType": "image/png"},
        {"type": "resource", "resource": {...}}
    ]
}

# LLM feedback (simplified)
{
    "type": "tool_result",
    "content": "Result 1"  # Text only
}
```

**理由:**
- 現在のLLM APIは画像/リソースをtool_resultとしてサポートしていない
- テキストのみに制限することで実装を単純化
- 将来的にマルチモーダル対応時に拡張可能

### 3. 非同期 vs 同期
`execute_with_tools()`は非同期ジェネレーター（`async def ... -> AsyncGenerator`）として実装。

**理由:**
- MCPClientが非同期API（`async def call_tool()`）
- ストリーミングレスポンスの維持（リアルタイムUI更新）
- Web UIのGradioも非同期対応

**影響:**
- CLI/Web UIの統合コードも非同期化が必要
- 既存の同期コードは`asyncio.run()`でラップ

### 4. ループ制御パラメータ
デフォルト値の根拠：

| パラメータ | デフォルト | 理由 |
|------------|------------|------|
| `max_iterations` | 10 | OpenAI Assistants APIの推奨値（参考: https://platform.openai.com/docs/assistants/tools/function-calling） |
| `timeout` | 120秒 | 複雑なツール呼び出しチェーン（例: 検索→分析→レポート生成）を考慮 |

**設計のトレードオフ:**
- 小さい値: レスポンス速度向上、無限ループ防止
- 大きい値: 複雑なタスク対応、柔軟性

### 5. エラーハンドリング戦略
ツール実行失敗時、**LLMにエラー内容を伝達**してリカバリーを試みる。

```python
try:
    result = await mcp_client.call_tool(name, arguments)
except Exception as e:
    # Don't raise - let LLM handle the error
    tool_result = {"name": name, "content": f"Tool execution failed: {e}"}
    yield {"type": "tool_result", "content": tool_result}
```

**理由:**
- LLMが代替手段を提案できる（例: "検索に失敗したので別のキーワードで試す"）
- ユーザーに対してより親切なエラーメッセージ
- 一部のツール失敗が全体のフローを止めない

**例外:**
- `ConnectionError`（MCPサーバーダウン）は即座に`raise`

---

## セキュリティ考慮事項

### 1. 無限ループ防止
- `max_iterations`で強制終了
- 各イテレーションで`timeout`監視

### 2. ツール実行の制限
- MCPサーバーが提供するツールのみ実行可能（`list_tools()`で取得）
- 任意のコード実行は不可

### 3. 入力バリデーション
- ツール引数の型検証はMCPサーバー側で実施
- アプリ側は`inputSchema`を信頼

### 4. エラー情報の露出
- ツール実行エラーをLLMに伝達する際、スタックトレース全体は含めない
- `str(e)`のみを使用（機密情報漏洩防止）

---

## パフォーマンス最適化

### 1. ツール定義のキャッシュ
```python
# list_tools()をループ外で1回だけ実行
tools = await mcp_client.list_tools()
for iteration in range(max_iterations):
    provider.call_api(history, system_prompt, tools)  # Reuse
```

### 2. 並列ツール実行（将来拡張）
現在の実装は**順次実行**だが、将来的に`asyncio.gather()`で並列化可能：

```python
# Current (sequential)
for tool_call in tool_calls_in_turn:
    result = await mcp_client.call_tool(...)

# Future (parallel)
results = await asyncio.gather(*[
    mcp_client.call_tool(tc["name"], tc["arguments"])
    for tc in tool_calls_in_turn
])
```

**注意:** 並列実行はツール間の依存関係がない場合のみ有効。

---

## 制限事項

### 現時点の制限
1. **マルチモーダル tool_result 非対応**
   - 画像/リソースはテキストに変換またはスキップ
   - 将来のLLM API拡張待ち

2. **並列ツール実行なし**
   - 順次実行のみ（実装の単純化のため）
   - パフォーマンスが問題になれば Phase 2 で対応

3. **ツール選択の制御なし**
   - LLMが`tool_choice="auto"`で自動判断
   - 特定ツールの強制実行は未サポート

4. **ストリーミング中断不可**
   - ユーザーがループを途中で止められない
   - CLI: Ctrl+C で強制終了のみ
   - Web UI: Gradioの"Stop"ボタンで中断（実装次第）

### 将来的な拡張候補
- [ ] 並列ツール実行（`asyncio.gather()`）
- [ ] `tool_choice`パラメータの公開
- [ ] ストリーミング中断機能（キャンセレーショントークン）
- [ ] マルチモーダル tool_result 対応
- [ ] ツール実行のロギング/デバッグ機能

---

## 受け入れ条件

### 機能要件
- [ ] `MCPClient.call_tool()`が実装され、ツールを実行できる
- [ ] `execute_with_tools()`がAgentic Loopを実装し、複数回のツール呼び出しをサポート
- [ ] CLIでツール呼び出しと結果が視覚的に表示される
- [ ] Web UIでツール呼び出しと結果がMarkdown形式で表示される
- [ ] 最大反復回数に達したら警告を出してループを終了する
- [ ] タイムアウトを超えたら`TimeoutError`を送出する
- [ ] ツール実行失敗時、エラー内容をLLMにフィードバックする

### テスト要件
- [ ] `tests/test_mcp_client.py`に2つのテストケース追加（成功/失敗）
- [ ] `tests/test_agentic_loop.py`を新規作成（4つのテストケース）
  - 正常系: 単一イテレーション
  - 境界値: max_iterations到達
  - 異常系: タイムアウト
  - 異常系: ツール実行エラー
- [ ] `tests/test_cli.py`に1つのテストケース追加（ツール表示）
- [ ] `tests/test_webui_handlers.py`に1つのテストケース追加（ツール表示）
- [ ] 全テスト（260+ tests）が通過
- [ ] Ruff lint/formatが通過

### ドキュメント要件
- [ ] `doc/`にAgentic Loop利用ガイドを追加
  - 設定方法（`max_iterations`, `timeout`）
  - ツール実行フロー図
  - トラブルシューティング（無限ループ、タイムアウト）
- [ ] READMEにAgentic Loop機能を追記

---

## 参考資料

- **先行実装:**
  - Issue #79（Gemini tools）: PR #94
  - Issue #80（ChatGPT tools）: PR #97
- **MCP仕様:**
  - Tool execution: https://modelcontextprotocol.io/docs/concepts/tools
  - CallToolResult: https://modelcontextprotocol.io/docs/specification/server/tools
- **OpenAI Assistants API:**
  - Function calling loop: https://platform.openai.com/docs/assistants/tools/function-calling
- **LangChain AgentExecutor:**
  - Similar loop implementation: https://python.langchain.com/docs/modules/agents/agent_types/

---

## 次のステップ（Issue #81完了後）

1. **Issue #82**: Web UI でのMCPサーバー設定UI
   - サーバー接続/切断の動的制御
   - 利用可能なツール一覧の表示

2. **Issue #83**: Filesystem MCPサーバー統合
   - `filesystem` MCPサーバーの接続
   - ファイル読み取りのE2Eテスト

3. **Story #78完了**: 全体的な統合テストとドキュメント整備
