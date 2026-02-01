# 構造化コンテンツ形式仕様書

## 概要

Issue #79のツールサポート実装に伴い、履歴エントリの`content`フィールドを文字列からリスト形式に拡張しました。これにより、テキスト、ツール呼び出し、ツール結果を統一的に扱えるようになりました。

## 構造化コンテンツ形式

### 基本構造

各履歴エントリは以下の形式を持ちます：

```python
{
    "role": str,  # "user", "gemini", "chatgpt", "tool"
    "content": List[Dict[str, Any]] | str  # 構造化リストまたは文字列（後方互換性）
}
```

### コンテンツパートの型

`content`リストには、以下の3種類のパートを含めることができます：

#### 1. テキストパート

```python
{
    "type": "text",
    "content": str  # 実際のテキストコンテンツ
}
```

**例:**
```python
{
    "role": "user",
    "content": [
        {"type": "text", "content": "Hello, how are you?"}
    ]
}
```

#### 2. ツール呼び出しパート

```python
{
    "type": "tool_call",
    "content": {
        "name": str,         # ツール名
        "arguments": dict,   # ツール引数
        "tool_call_id": str  # (オプション) ツール呼び出しID
    }
}
```

**例:**
```python
{
    "role": "gemini",
    "content": [
        {"type": "text", "content": "Let me check the weather for you."},
        {
            "type": "tool_call",
            "content": {
                "name": "get_weather",
                "arguments": {"location": "Tokyo"},
                "tool_call_id": "call_abc123"
            }
        }
    ]
}
```

#### 3. ツール結果パート

```python
{
    "type": "tool_result",
    "name": str,           # ツール名
    "content": str | dict, # ツール実行結果
    "tool_call_id": str    # (オプション) 対応するツール呼び出しID
}
```

**例:**
```python
{
    "role": "tool",
    "content": [
        {
            "type": "tool_result",
            "name": "get_weather",
            "content": '{"temperature": "25°C", "condition": "sunny"}',
            "tool_call_id": "call_abc123"
        }
    ]
}
```

## 各プロバイダでのマッピング

### GeminiProvider

#### 入力（アプリケーション→Gemini API）

構造化コンテンツをGemini API形式に変換（`GeminiProvider.format_history`）：

| アプリケーション形式 | Gemini API形式 |
|-------------------|---------------|
| `role: "user"` | `role: "user"` |
| `role: "gemini"` | `role: "model"` |
| `role: "tool"` | `role: "function"` |
| `type: "text"` | `parts: [{"text": str}]` |
| `type: "tool_call"` | `parts: [{"function_call": {"name": str, "args": dict}}]` |
| `type: "tool_result"` | `parts: [{"function_response": {"name": str, "response": dict, "id": str}}]` |

#### 出力（Gemini API→アプリケーション）

Gemini APIのストリーミングレスポンスを統一形式に変換（`GeminiProvider.call_api`）：

| Gemini API形式 | 統一辞書形式 |
|---------------|-------------|
| `chunk.parts[].text` | `{"type": "text", "content": str}` |
| `chunk.parts[].function_call` | `{"type": "tool_call", "content": {"name": str, "arguments": dict}}` |

**特殊処理:**
- ツール名と引数が別々のチャンクで届く場合があるため、`GeminiToolCallAssembler`クラスで組み立て
- 空の引数（`{}`）は無視し、実際の引数が到着した時点でツール呼び出しイベントを発火

### ChatGPTProvider

#### 入力（アプリケーション→OpenAI API）

構造化コンテンツをOpenAI API形式に変換（`ChatGPTProvider.format_history`）：

| アプリケーション形式 | OpenAI API形式 |
|-------------------|--------------|
| `role: "user"` | `role: "user"` |
| `role: "chatgpt"` | `role: "assistant"` |
| `role: "tool"` | `role: "tool"` |
| `type: "text"` | `content: str` |
| `type: "tool_call"` | `tool_calls: [{"id": str, "type": "function", "function": {...}}]` |
| `type: "tool_result"` | `content: str, tool_call_id: str` |

#### 出力（OpenAI API→アプリケーション）

OpenAI APIのストリーミングレスポンスを統一形式に変換（`ChatGPTProvider.call_api`）：

| OpenAI API形式 | 統一辞書形式 |
|---------------|-------------|
| `chunk.choices[0].delta.content` | `{"type": "text", "content": str}` |
| `chunk.choices[0].delta.tool_calls[i].function` | `{"type": "tool_call", "content": {"name": str, "arguments": dict, "tool_call_id": str}}` |

**Note:** ツール結果の返送時は、統一形式の`role: "tool"`をOpenAI API形式の`role: "tool", tool_call_id: str`に変換します（`ChatGPTProvider.format_history`内で処理）。

## 後方互換性

### レガシー文字列形式

既存コードとの互換性のため、`content`が文字列の場合も許容されます：

```python
# レガシー形式
{
    "role": "user",
    "content": "Hello"
}

# 内部的に以下に変換
{
    "role": "user",
    "content": [{"type": "text", "content": "Hello"}]
}
```

### 履歴の正規化

`history_utils.normalize_history_turns`関数により、レガシー形式の履歴を新しい構造化形式に変換：

```python
from multi_llm_chat.history_utils import normalize_history_turns

legacy_history = [
    {"role": "user", "content": "Hello"},
    {"role": "gemini", "content": "Hi there"}
]

normalized = normalize_history_turns(legacy_history)
# => [
#     {"role": "user", "content": [{"type": "text", "content": "Hello"}]},
#     {"role": "gemini", "content": [{"type": "text", "content": "Hi there"}]}
# ]
```

## ユーティリティ関数

### content_to_text

構造化コンテンツをプレーンテキストに変換（トークン計算やUI表示用）：

```python
from multi_llm_chat.history_utils import content_to_text

content = [
    {"type": "text", "content": "Let me search."},
    {"type": "tool_call", "content": {"name": "search", "arguments": {"query": "python"}}},
    {"type": "text", "content": "Here are the results."}
]

# ツールデータを含めない（UI表示用）
text = content_to_text(content, include_tool_data=False)
# => "Let me search. Here are the results."

# ツールデータを含める（トークン計算用）
text = content_to_text(content, include_tool_data=True)
# => "Let me search. search {\"query\":\"python\"} Here are the results."
```

## UI表示の処理

### ストリーミング中の表示

`ChatService.process_message`では、ツール呼び出しを以下のように表示用履歴に追加：

```python
# ツール呼び出しを検出した場合
tool_name = tool_call_content.get("name", "unknown_tool")
tool_representation = f"[Tool Call: {tool_name}]"
self.display_history[-1][1] += tool_representation
```

### 履歴ロード時の表示

`webui/handlers.logic_history_to_display`や`cli`では、`content_to_text(include_tool_data=False)`を使用してテキストのみを抽出して表示します。

**注意:** 現在の実装では、履歴ロード時にツール呼び出し情報が表示されません。これは将来の改善課題です（修正設計書の「優先度3」参照）。

## エラーハンドリング

### 無効なコンテンツ型

`format_history`は、`content`がリストでも文字列でもない場合に`ValueError`を発生させます：

```python
history = [{"role": "user", "content": {"text": "bad"}}]

# ValueError: History content must be a list or string (actual type: dict)
```

### ツール結果のJSON解析

`_parse_tool_response_payload`関数は、ツール結果を柔軟に解析：

1. `None` → `{}`
2. 辞書 → そのまま
3. JSON文字列 → パース（失敗時は`{"result": str}`でラップ）
4. その他 → `{"result": value}`でラップ

## テストパターン

### 構造化コンテンツのモック

テストでは、統一辞書形式でモックを作成：

```python
mock_provider.call_api.return_value = iter([
    {"type": "text", "content": "Hello"},
    {"type": "tool_call", "content": {"name": "search", "arguments": {"q": "test"}}}
])
```

### Gemini APIレスポンスのモック

Gemini APIの生のレスポンス形式をモック：

```python
mock_part = MagicMock(spec=["text"])  # function_call属性を持たない
mock_part.text = "Response text"
mock_chunk = MagicMock()
mock_chunk.parts = [mock_part]
mock_model.generate_content.return_value = iter([mock_chunk])
```

**重要:** `spec=["text"]`により、`function_call`属性が存在しないことを明示的に指定します。これにより、`call_api`がテキストとして正しく処理します。

## 関連ファイル

- `src/multi_llm_chat/llm_provider.py` - プロバイダ実装と形式変換
- `src/multi_llm_chat/history_utils.py` - 履歴ユーティリティ関数
- `src/multi_llm_chat/chat_logic.py` - ChatServiceの実装
- `tests/test_gemini_tools.py` - ツール機能の統合テスト
- `tests/test_llm_provider.py` - プロバイダのユニットテスト
