# Issue #80 設計サマリー

## 主要な設計決定

### 1. アーキテクチャの統一
- **Gemini実装（Issue #79）との一貫性を維持**
  - 変換関数: `mcp_tools_to_openai_format()` (Geminiは`mcp_tools_to_gemini_format()`)
  - パース関数: `parse_openai_tool_call()` (Geminiは`parse_gemini_function_call()`)
  - Assemblerクラス: `OpenAIToolCallAssembler` (Geminiは`GeminiToolCallAssembler`)

### 2. OpenAI固有の特徴への対応

#### A. ツール呼び出しID
```python
# OpenAIはtool_call_idが必須（Geminiにはない）
{
    "tool_name": "get_weather",
    "arguments": {"location": "Tokyo"},
    "tool_call_id": "call_abc123"  # ← OpenAI固有
}
```

#### B. JSON文字列引数
```python
# OpenAI: arguments は JSON文字列
"arguments": "{\"location\": \"Tokyo\"}"  # json.loads()が必要

# Gemini: arguments は既にdictオブジェクト
"arguments": {"location": "Tokyo"}  # そのまま使用可能
```

#### C. ストリーミングの振る舞い
```
OpenAI:
  Chunk 1: {index: 0, id: "call_123", function: {name: "get_weather"}}
  Chunk 2: {index: 0, function: {arguments: "{\"loc"}}
  Chunk 3: {index: 0, function: {arguments: "ation\": \"T"}}
  Chunk 4: {index: 0, function: {arguments: "okyo\"}"}}

Gemini:
  Chunk 1: {function_call: {name: "get_weather"}}
  Chunk 2: {function_call: {args: {"location": "Tokyo"}}}
```

### 3. 共通形式の統一（修正版）

#### 実装の決定: フィールド名は "name" で統一

**基本形式（Gemini/OpenAI共通）**:
- `name`: ツール名（文字列）
- `arguments`: ツール引数（辞書）

**OpenAI固有フィールド**:
- `tool_call_id`: ツール呼び出しID（文字列、tool_resultメッセージで使用）

```python
# Geminiの戻り値:
{"name": "get_weather", "arguments": {"location": "Tokyo"}}

# OpenAIの戻り値:
{
    "name": "get_weather",
    "arguments": {"location": "Tokyo"},
    "tool_call_id": "call_abc123"  # OpenAI固有（Geminiでは None）
}
```

**設計判断の理由**:
1. Gemini実装が既に `"name"` キーを使用している
2. フィールド名を統一することでUI層（CLI/WebUI）のハンドリングがシンプルになる
3. `tool_call_id` はOpenAI固有の追加フィールドとして扱う（Geminiでは `None`）

### 4. format_history()の拡張

#### 現状の問題
```python
# 現在はツール呼び出しをテキストに変換してしまう
chatgpt_history.append({"role": "assistant", "content": content_to_text(content)})
```

#### 修正案
```python
# 構造化コンテンツを保持
# OpenAI API specification:
# - content and tool_calls can coexist (mixed content is valid)
# - content should be None only when no text is present
if text_items or tool_call_items:
    message = {"role": "assistant"}
    if text_items:
        message["content"] = " ".join(text_items)
    if tool_call_items:
        message["tool_calls"] = [...]  # Convert format
    # Only set content=None if we have tool_calls but NO text
    if tool_call_items and not text_items:
        message["content"] = None
    chatgpt_history.append(message)
```

### 5. エラーハンドリング戦略

#### A. 不完全なJSON引数
```python
# OpenAIはargumentsを段階的に送信
# 最後のチャンクまで待ってからjson.loads()を実行

class OpenAIToolCallAssembler:
    def process_tool_call(self, tool_call_delta):
        # arguments_jsonを蓄積
        self._tools_by_index[index]["arguments_json"] += function.arguments
        
        # ストリーム終了時またはfinish_reasonで完了判定
```

#### B. パース失敗時の対応
```python
try:
    arguments = json.loads(args_json)
except json.JSONDecodeError as e:
    logger.warning("Failed to parse tool arguments JSON: %s", e)
    arguments = {}  # 空のdictとして続行（graceful degradation）
```

## 実装の優先順位

### High Priority（Phase 1-2）
1. ✅ 変換関数（`mcp_tools_to_openai_format`, `parse_openai_tool_call`）
2. ✅ `OpenAIToolCallAssembler`クラス
3. ✅ `call_api()`の拡張（toolsパラメータ対応）
4. ✅ ストリーミング処理の統合

### Medium Priority（Phase 3）
5. ✅ `format_history()`の構造化コンテンツ対応
6. ✅ トークン計算の更新（buffer factorの適用）

### Low Priority（Phase 4）
7. 🔄 エラーケースのテスト追加
8. 🔄 ロギングとデバッグ用の情報追加

## テストケース一覧

### 変換関数（4テスト）
1. `test_mcp_to_openai_tool_conversion` - 正常な変換
2. `test_mcp_to_openai_with_empty_tools` - 空配列/None処理
3. `test_parse_openai_tool_call` - 完全な構造のパース
4. `test_parse_openai_tool_call_with_invalid_json` - 不正JSON処理

### Provider統合（4テスト）
5. `test_chatgpt_provider_call_api_with_tools` - tools引数の伝播
6. `test_chatgpt_response_with_tool_call` - tool_call検出
7. `test_chatgpt_streaming_tool_arguments` - 段階的arguments組み立て
8. `test_chatgpt_parallel_tool_calls` - 並列呼び出し

### エラーケース（3テスト）
9. `test_invalid_tool_arguments_json` - 不正JSON
10. `test_missing_tool_call_id` - idフィールド欠落
11. `test_tool_call_without_name` - nameフィールド欠落

## 注意点とベストプラクティス

### 1. Geminiとの差異を意識
- OpenAIはJSON文字列、Geminiはdictオブジェクト
- OpenAIにはtool_call_id、Geminiにはない
- エラーハンドリングをそれぞれ適切に実装

### 2. ストリーミングのロバスト性
- arguments_jsonは完全に蓄積されるまでパースしない
- index-based trackingで並列呼び出しに対応
- ストリーム中断時のクリーンアップを忘れずに

### 3. 後方互換性の維持
- `tools=None`時は既存の動作を保持
- 古い履歴形式もcontent_to_text()でフォールバック
- エラーメッセージを適切に更新

### 4. TDDサイクルの徹底
- 11テスト全てを先に作成（Red）
- Phase 1-4で段階的に実装（Green）
- 各Phase完了後にリファクタリング（Refactor）

## 実装後の検証項目

- [ ] 全232+11=243テストがパス
- [ ] GeminiとChatGPTの両方でツール呼び出しが動作
- [ ] 並列ツール呼び出しが正しく処理される
- [ ] 履歴にツール呼び出しが正しく保存される
- [ ] トークン計算がツール有無で適切に調整される
- [ ] CI（Python 3.10, 3.11）が成功

## 次のステップへの準備

Issue #80完了後、Story #78の次のタスクに進む準備:
- ツール実行ループ（Agentic Loop）の実装
- MCPサーバーとの統合テスト
- エンドツーエンドのツール実行検証
