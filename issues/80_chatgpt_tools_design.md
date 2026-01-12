# Issue #80: ChatGPTProviderにおけるtoolsパラメータのマッピング実装

## 概要

ChatGPTProviderを拡張し、MCPツール定義をOpenAI API形式に変換して関数呼び出しをサポートする。

## 設計方針

### 1. Geminiとの設計統一性

Issue #79で実装されたGeminiProviderのツール対応と同様のアーキテクチャを採用:
- MCP形式からOpenAI形式への変換関数を提供
- ストリーミングレスポンスでのツール呼び出しを統一形式で返す
- 既存の`call_api()`シグネチャを維持（後方互換性）

### 2. OpenAI Tools API 仕様

#### ツール定義形式（リクエスト）
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"  // "auto" | "none" | {"type": "function", "function": {"name": "..."}}
}
```

#### ツール呼び出し形式（レスポンス）
```json
{
  "choices": [
    {
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Tokyo\"}"
            }
          }
        ]
      }
    }
  ]
}
```

**ストリーミング時の特徴:**
- `tool_calls`は複数チャンクに分割されて送信される
- 各チャンクに`index`フィールドがあり、並列ツール呼び出しを識別
- `arguments`はJSON文字列として段階的に送信される（`{"loc`→`ation": "T`→`okyo"}`）
- Geminiと異なり、`id`フィールドが存在する（後続のtool_resultメッセージで使用）

### 3. MCP形式との変換

#### MCP → OpenAI変換
```python
def mcp_tools_to_openai_format(mcp_tools: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Convert MCP tool definitions to OpenAI tools format.
    
    Args:
        mcp_tools: [{"name": str, "description": str, "inputSchema": dict}, ...]
    
    Returns:
        [{"type": "function", "function": {"name": str, "description": str, "parameters": dict}}, ...]
    """
```

**マッピング:**
- MCP `name` → OpenAI `function.name`
- MCP `description` → OpenAI `function.description`
- MCP `inputSchema` → OpenAI `function.parameters`

#### OpenAI → 共通形式への変換
```python
def parse_openai_tool_call(tool_call: dict) -> Dict[str, Any]:
    """Parse OpenAI tool_call to common format.
    
    Args:
        tool_call: {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
    
    Returns:
        {"tool_name": str, "arguments": dict, "tool_call_id": str}
    """
```

**特記事項:**
- OpenAIの`arguments`はJSON文字列なので`json.loads()`が必要
- `id`フィールドを保持（tool_resultメッセージ構築に必要）
- Geminiの共通形式は`{"name": str, "arguments": dict}`だが、OpenAIでは`tool_call_id`も含める

## 実装計画

### Phase 1: 変換関数の実装（TDD）

**テストケース（tests/test_chatgpt_tools.py）:**
1. `test_mcp_to_openai_tool_conversion`: MCP→OpenAI形式変換の正確性
2. `test_mcp_to_openai_with_empty_tools`: 空配列/None処理
3. `test_parse_openai_tool_call`: 完全なtool_call構造のパース
4. `test_parse_openai_tool_call_with_invalid_json`: 不正なJSON argumentsのエラーハンドリング

**実装（src/multi_llm_chat/llm_provider.py）:**
```python
# Add after mcp_tools_to_gemini_format()

def mcp_tools_to_openai_format(mcp_tools: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Convert MCP tool definitions to OpenAI tools format."""
    if not mcp_tools:
        return None
    
    openai_tools = []
    for tool in mcp_tools:
        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("inputSchema")
        
        if not name:
            logger.warning("Skipping MCP tool without name.")
            continue
        
        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description or "",
                "parameters": parameters or {}
            }
        })
    
    return openai_tools if openai_tools else None


def parse_openai_tool_call(tool_call: dict) -> Dict[str, Any]:
    """Parse OpenAI tool_call to common format."""
    function = tool_call.get("function", {})
    name = function.get("name")
    args_json = function.get("arguments", "{}")
    tool_call_id = tool_call.get("id")
    
    # Parse JSON arguments
    try:
        arguments = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse tool arguments JSON: %s", e)
        arguments = {}
    
    return {
        "tool_name": name,
        "arguments": arguments,
        "tool_call_id": tool_call_id
    }
```

### Phase 2: ChatGPTProviderの拡張（TDD）

**テストケース（tests/test_chatgpt_tools.py）:**
5. `test_chatgpt_provider_call_api_with_tools`: tools引数がAPI呼び出しに含まれること
6. `test_chatgpt_response_with_tool_call`: ストリーミング中のtool_call検出
7. `test_chatgpt_streaming_tool_arguments`: 複数チャンクに分割されたargumentsの組み立て
8. `test_chatgpt_parallel_tool_calls`: 並列ツール呼び出し（複数index）の処理

**実装の焦点:**

#### A. ストリーミング中のツール呼び出し組み立て

OpenAI APIの特性:
- `tool_calls`配列の各要素に`index`がある（並列呼び出し用）
- `arguments`は段階的に送信される（`"{"` → `"location"` → `": \"Tokyo\""` → `"}"`）
- `id`と`name`は最初のチャンクで送信される

**実装方針:**
```python
class OpenAIToolCallAssembler:
    """Assembles OpenAI streaming tool calls.
    
    State:
        _tools_by_index: Dict[int, Dict]
            Key: tool_call.index
            Value: {
                "id": str,
                "name": str, 
                "arguments_json": str,  # Accumulated JSON string
                "complete": bool
            }
    """
    
    def process_tool_call(self, tool_call_delta):
        """Process streaming tool_call delta.
        
        Args:
            tool_call_delta: {
                "index": int,
                "id": str (optional, first chunk only),
                "function": {
                    "name": str (optional, first chunk only),
                    "arguments": str (partial JSON fragment)
                }
            }
        
        Returns:
            Optional[Dict]: Completed tool_call in common format, or None
        """
```

**Geminiとの違い:**
- Gemini: 名前とargsが別チャンクで送信される場合がある
- OpenAI: 名前は最初のチャンクのみ、argumentsは段階的に蓄積
- OpenAI: JSON文字列なので`json.loads()`が必要（パース失敗に注意）
- OpenAI: `id`フィールドがある

#### B. call_api()の修正

**既存コード（line 696-713）:**
```python
def call_api(self, history, system_prompt=None, tools: Optional[List[Dict[str, Any]]] = None):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    if tools:
        raise ValueError(
            "ChatGPTProvider does not support tools yet. "
            "Use @gemini mention or wait for future implementation."
        )
    # ... existing implementation
```

**修正後:**
```python
def call_api(self, history, system_prompt=None, tools: Optional[List[Dict[str, Any]]] = None):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    
    # Build messages
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    
    chatgpt_history = self.format_history(history)
    messages.extend(chatgpt_history)
    
    # Prepare API call parameters
    api_params = {
        "model": CHATGPT_MODEL,
        "messages": messages,
        "stream": True
    }
    
    # Add tools if provided
    if tools:
        openai_tools = mcp_tools_to_openai_format(tools)
        if openai_tools:
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"  # Let model decide when to call
    
    # Call API
    stream = self._client.chat.completions.create(**api_params)
    
    # Process stream with tool call assembler
    assembler = OpenAIToolCallAssembler()
    for chunk in stream:
        # Check for tool calls
        if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
            for tool_call_delta in chunk.choices[0].delta.tool_calls:
                result = assembler.process_tool_call(tool_call_delta)
                if result:
                    yield {"type": "tool_call", "content": result}
        
        # Check for text content
        text_content = self.extract_text_from_chunk(chunk)
        if text_content:
            yield {"type": "text", "content": text_content}
```

#### C. format_history()の更新

**現状（line 668-694）:**
```python
# TODO: When adding tool support, preserve tool_call and tool_result
# in structured format instead of flattening to text
chatgpt_history.append({"role": "assistant", "content": content_to_text(content)})
```

**修正案:**
```python
elif role == "chatgpt":
    # Preserve structured content for tool support
    if isinstance(content, dict):
        # Handle tool_call/tool_result structured format
        if content.get("type") == "tool_call":
            # Convert to OpenAI tool_calls format
            chatgpt_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": content["tool_call_id"],
                    "type": "function",
                    "function": {
                        "name": content["tool_name"],
                        "arguments": json.dumps(content["arguments"])
                    }
                }]
            })
        elif content.get("type") == "tool_result":
            # Convert to OpenAI tool message format
            chatgpt_history.append({
                "role": "tool",
                "tool_call_id": content["tool_call_id"],
                "content": json.dumps(content["result"])
            })
        else:
            # Plain text in dict format
            chatgpt_history.append({"role": "assistant", "content": content_to_text(content)})
    else:
        # Legacy string format
        chatgpt_history.append({"role": "assistant", "content": content_to_text(content)})
```

### Phase 3: トークン計算の更新

**考慮事項:**
- OpenAI toolsはリクエストサイズを増加させる
- Geminiと同様にトークンバッファファクターを調整（1.5）

**実装（src/multi_llm_chat/token_utils.py）:**
```python
# get_buffer_factor() already supports has_tools parameter (Issue #79)
# No changes needed
```

**実装（ChatGPTProvider.get_token_info()）:**
```python
def get_token_info(self, text, history=None, model_name=None, has_tools=False):
    # ... existing tiktoken logic ...
    
    # Apply buffer factor
    buffer_factor = get_buffer_factor(has_tools=has_tools)
    token_count = int(token_count * buffer_factor)
    
    # ... rest of logic
```

### Phase 4: エラーハンドリングとロバスト性

**テストケース:**
9. `test_invalid_tool_arguments_json`: 不正なJSON arguments
10. `test_missing_tool_call_id`: idフィールドが欠けている場合
11. `test_tool_call_without_name`: 名前のないツール呼び出し

**実装の焦点:**
- JSON parse失敗時のgraceful degradation
- 部分的に受信したツール呼び出しの処理
- ストリーム中断時のクリーンアップ

## テスト戦略

### ユニットテスト（tests/test_chatgpt_tools.py）
- 11テストケースを追加
- モックを使用してAPI呼び出しをシミュレート
- エッジケースとエラーケースをカバー

### 統合テスト（tests/test_llm_provider.py）
- 既存の`TestChatGPTProvider`クラスに追加
- GeminiとChatGPTの動作一貫性を検証

## マイグレーション考慮事項

### 後方互換性
- `tools`パラメータは`Optional`なので既存コードは影響なし
- `tools=None`時は現在の動作を維持
- エラーメッセージ変更（"does not support tools yet" → ツール対応済み）

### 履歴フォーマット
- `format_history()`の変更により、過去の履歴もツール対応形式で処理可能
- `content_to_text()`フォールバックにより古い履歴との互換性維持

## 実装順序（TDD）

1. **Red**: 11テストケースを作成（全て失敗）
2. **Green Phase 1**: 変換関数を実装（テスト1-4がパス）
3. **Green Phase 2**: OpenAIToolCallAssemblerを実装（テスト5-8がパス）
4. **Green Phase 3**: call_api()とformat_history()を更新（全テストパス）
5. **Refactor**: コードクリーンアップ、ログ追加、ドキュメント改善

## 完了条件

- [ ] 全11テストケースがパス
- [ ] 既存のChatGPTProviderテスト（tests/test_llm_provider.py）が全てパス
- [ ] CI（Python 3.10, 3.11）が成功
- [ ] Ruff lint/formatチェックがパス
- [ ] README.mdにツール対応を記載（オプション）

## 参考資料

- OpenAI Function Calling API: https://platform.openai.com/docs/guides/function-calling
- Issue #79実装（Gemini tools）: PR #94
- MCP仕様: https://modelcontextprotocol.io/

## リスクと制限事項

### リスク
1. OpenAI APIの`arguments`が不完全なJSON断片で送信される
   - 対策: 完全なJSONが揃うまでバッファリング、終了時にパース
2. 並列ツール呼び出しの`index`順序が保証されない可能性
   - 対策: indexベースの辞書管理、順序に依存しない実装

### 制限事項
1. `tool_choice`は常に`"auto"`（モデルが自動判断）
   - 将来的に設定可能にすることも検討可能
2. ツール実行結果のフィードバック（tool_resultの送信）はStory #78の別タスクで実装

## 次のステップ（Issue #80完了後）

- Issue #81: ツール実行ループの実装（Agentic Loop）
- Story #78全体の完了確認
