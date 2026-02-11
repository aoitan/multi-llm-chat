# Deprecation Policy - Legacy API

## 概要

本ドキュメントでは、`multi_llm_chat`プロジェクトにおけるLegacy API（非推奨API）の一覧、削除予定時期、および移行ガイドを提供します。

## Legacy API一覧

以下の関数は`core_modules.legacy_api`モジュールに含まれており、**非推奨（DEPRECATED）**として扱われます。

### 同期API関数

| 関数名 | 説明 | 非推奨理由 |
|--------|------|-----------|
| `call_gemini_api()` | Gemini APIへの同期呼び出し | Providerパターンへの移行推奨 |
| `call_chatgpt_api()` | ChatGPT APIへの同期呼び出し | Providerパターンへの移行推奨 |
| `call_gemini_api_async()` | Gemini APIへの非同期呼び出し | Providerパターンへの移行推奨 |
| `call_chatgpt_api_async()` | ChatGPT APIへの非同期呼び出し | Providerパターンへの移行推奨 |

### ユーティリティ関数

| 関数名 | 説明 | 非推奨理由 |
|--------|------|-----------|
| `format_history_for_gemini()` | Gemini形式への履歴変換 | Provider内部で自動実行 |
| `format_history_for_chatgpt()` | ChatGPT形式への履歴変換 | Provider内部で自動実行 |
| `prepare_request()` | リクエスト準備ヘルパー | `history_utils.prepare_request()`を直接使用 |
| `load_api_key()` | 環境変数からAPIキーを読み込み | `llm_provider.load_api_key()`を直接使用 |
| `stream_text_events()` | テキストイベントのストリーミング | Provider APIを使用 |
| `stream_text_events_async()` | 非同期テキストストリーミング | Provider APIを使用 |
| `extract_text_from_chunk()` | チャンクからテキスト抽出 | Provider APIを使用 |

## Deprecationタイムライン

| 時期 | アクション |
|------|-----------|
| **現在 (v1.x)** | DeprecationWarning表示、`core.py`の`__all__`から除外 |
| **v2.0.0** | Legacy API削除予定、破壊的変更としてリリース |

## 推奨される移行パス

### 1. 旧APIから新Provider APIへの移行

**Before (Legacy API):**
```python
from multi_llm_chat.core import call_gemini_api

response = call_gemini_api(
    history=[{"role": "user", "content": "Hello"}],
    model="gemini-2.0-flash-exp",
    stream=False
)
```

**After (Provider API):**
```python
from multi_llm_chat.llm_provider import create_provider

provider = create_provider("gemini")
response = provider.call_api(
    history=[{"role": "user", "content": "Hello"}],
    model="gemini-2.0-flash-exp",
    stream=False
)
```

### 2. 履歴フォーマット変換の移行

**Before:**
```python
from multi_llm_chat.core import format_history_for_gemini

formatted = format_history_for_gemini(history)
```

**After:**
Provider内部で自動的に変換されるため、呼び出し不要です。Provider APIに直接履歴を渡してください。

### 3. APIキー読み込みの移行

**Before:**
```python
from multi_llm_chat.core import load_api_key

api_key = load_api_key("GOOGLE_API_KEY")
```

**After:**
```python
from multi_llm_chat.llm_provider import load_api_key

api_key = load_api_key("GOOGLE_API_KEY")
```

### 4. Agentic Loopの使用（推奨）

ツール呼び出しが必要な場合は、Agentic Loop APIを使用してください：

```python
from multi_llm_chat.core import execute_with_tools_stream

async for event in execute_with_tools_stream(
    history=history,
    model="gemini-2.0-flash-exp",
    tools=my_tools,
    max_iterations=5
):
    if event["type"] == "text":
        print(event["content"], end="", flush=True)
```

## Legacy APIの利用を続ける場合

やむを得ずLegacy APIを使い続ける必要がある場合は、以下の方法で明示的にインポートしてください：

```python
from multi_llm_chat.core_modules.legacy_api import call_gemini_api, call_chatgpt_api
```

この方法では、実行時に`DeprecationWarning`が表示されます。v2.0.0でこれらの関数は削除される予定です。

## 質問・フィードバック

Legacy APIの移行について質問や提案がある場合は、GitHubのIssueで報告してください。
