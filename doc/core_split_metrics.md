# core.py 分割リファクタリング - 成果メトリクス

## 概要

Issue #103の実装として、`core.py`（740行）を責務別に4つのモジュールへ分割し、保守性と可読性を向上させました。

**実施期間**: 2026-02-01  
**完了コミット**: 10個（準備2 + Phase 2: 4 + Phase 3: 3 + 最終検証1）

---

## Before / After 比較

### ファイル行数

| ファイル | Before | After | 変化 | 削減率 |
|---------|--------|-------|------|--------|
| `core.py` | 740行 | 129行 | ▼611行 | **82%削減** |
| `core_modules/__init__.py` | - | 13行 | +13行 | - |
| `core_modules/legacy_api.py` | - | 290行 | +290行 | - |
| `core_modules/token_and_context.py` | - | 201行 | +201行 | - |
| `core_modules/agentic_loop.py` | - | 423行 | +423行 | - |
| `core_modules/providers_facade.py` | - | 45行 | +45行 | - |
| **合計** | **740行** | **1,101行** | **+361行** | - |

**注記**: 行数増加はモジュールヘッダー、docstring、importの明示化によるもの。実装ロジックは変更なし。

### テスト状況

| 項目 | Before | After |
|------|--------|-------|
| tests/test_core.py | 28テスト | 28テスト |
| パス率 | 100% | 100% |
| テスト修正 | - | mock path修正（8箇所）、private関数import（3箇所） |

---

## モジュール構成

### Before: 単一モジュール（740行）

```
src/multi_llm_chat/
└── core.py (740行)
    ├── DEPRECATED wrappers (164行)
    ├── Token/validation wrappers (67行)
    ├── Agentic Loop (421行)
    ├── Provider utilities (80行)
    └── Constants & imports (8行)
```

**問題点**:
- 責務が混在（API wrapper, トークン計算, Agentic Loop, デバッグ）
- 変更影響範囲が不明確
- テスト対象の分離が困難
- 将来の拡張が難しい

### After: 責務分離（5モジュール, 1,101行）

```
src/multi_llm_chat/
├── core.py (129行) ← 純粋ファサード
└── core_modules/
    ├── __init__.py (13行)
    ├── legacy_api.py (290行) ← DEPRECATED wrappers
    ├── token_and_context.py (201行) ← Token & validation
    ├── agentic_loop.py (423行) ← Agentic Loop
    └── providers_facade.py (45行) ← Provider utilities
```

**改善点**:
- ✅ 責務が明確に分離
- ✅ 変更影響範囲が局所化
- ✅ テストが書きやすい
- ✅ 将来の拡張が容易（例: RAG, MCP拡張は agentic_loop.py に集約）

---

## 各モジュールの責務

### 1. `core.py` (129行) - Pure Facade

**役割**: 公開APIの統一窓口

**内容**:
- Import + re-export のみ
- `__all__` で公開API（44個）を明示
- 実装ロジックなし

**依存**: core_modules.*, llm_provider, history_utils

---

### 2. `core_modules/legacy_api.py` (290行)

**役割**: DEPRECATED backward compatibility wrappers

**主要関数**:
- `call_gemini_api()`, `call_gemini_api_async()` ← GeminiProvider委譲
- `call_chatgpt_api()`, `call_chatgpt_api_async()` ← ChatGPTProvider委譲
- `stream_text_events()`, `stream_text_events_async()` ← Provider委譲
- `format_history_for_gemini()` ← history_utils委譲
- `format_history_for_chatgpt()` ← history_utils委譲
- `extract_text_from_chunk()` ← history_utils委譲
- `prepare_request()` ← Provider委譲
- `load_api_key()` ← llm_provider委譲

**依存**: llm_provider, history_utils, openai

---

### 3. `core_modules/token_and_context.py` (201行)

**役割**: Token calculation & context validation wrappers

**主要関数**:
- `_estimate_tokens()` ← token_utils委譲
- `calculate_tokens()` ← token_utils委譲
- `get_token_info()` ← token_utils委譲
- `get_max_context_length()` ← llm_provider委譲
- `prune_history_sliding_window()` ← compression委譲
- `get_pruning_info()` ← compression委譲
- `validate_system_prompt_length()` ← validation委譲
- `validate_context_length()` ← validation委譲

**依存**: compression, validation, token_utils, llm_provider, history_utils

---

### 4. `core_modules/agentic_loop.py` (423行)

**役割**: Agentic Loop implementation（MCP tool execution）

**主要エンティティ**:
- `AgenticLoopResult` (dataclass) - 実行結果の不変データ構造
- `execute_with_tools_stream()` - ストリーミング実行（推奨API）
- `execute_with_tools()` - 非同期実行
- `execute_with_tools_sync()` - 同期実行（CLIなどで利用）

**特徴**:
- タイムアウト制御（デフォルト120秒）
- 最大イテレーション制御（デフォルト10回）
- MCP tool呼び出しハンドリング
- エラー処理と履歴管理

**依存**: history_utils.validate_history_entry, asyncio, logging

---

### 5. `core_modules/providers_facade.py` (45行)

**役割**: Provider access & debug utilities

**主要関数**:
- `list_gemini_models()` - 利用可能なGeminiモデル一覧（デバッグ用）

**依存**: google.generativeai, llm_provider

---

## 循環依存の回避戦略

**原則**: core_modules 内のモジュールは sibling modules に依存せず、parent package (`..llm_provider`, `..history_utils` など) のみに依存

**依存グラフ**:
```
core.py
  ↓ (re-export)
core_modules/*
  ↓ (delegate)
llm_provider, history_utils, token_utils, compression, validation
  ↓
External APIs (Gemini, ChatGPT)
```

---

## テスト修正内容

### 修正箇所

1. **mock path 変更**（8箇所）
   - Before: `patch("multi_llm_chat.core.create_provider")`
   - After: `patch("multi_llm_chat.core_modules.legacy_api.create_provider")`

2. **private関数 import 変更**（3箇所）
   - Before: `core._estimate_tokens(...)`
   - After: `from multi_llm_chat.core_modules.token_and_context import _estimate_tokens`

### テスト結果

```
tests/test_core.py: 28 passed (100%)
全283テスト: PASS
```

---

## 移行ガイド

### 既存コードへの影響

**影響なし** - 完全な後方互換性を維持

```python
# Before & After - 同じコードが動作
from multi_llm_chat.core import (
    call_gemini_api,
    execute_with_tools_stream,
    get_token_info,
)
```

### 新規コードでの推奨

将来的には、以下のように直接 import することを推奨:

```python
# 推奨（新規コード）
from multi_llm_chat.llm_provider import create_provider
from multi_llm_chat.core_modules.agentic_loop import execute_with_tools_stream

# 非推奨（DEPRECATED）
from multi_llm_chat.core import call_gemini_api
```

---

## 今後の改善提案

### Phase 4: DEPRECATED wrapper の段階的廃止

1. `call_gemini_api()` / `call_chatgpt_api()` の呼び出し元を `create_provider()` + `provider.call_api()` へ移行
2. `format_history_for_*()` の呼び出し元を Provider層へ統合
3. legacy_api.py のwarning強化（DeprecationWarning → FutureWarning）
4. 1-2リリース後に削除

### 将来の拡張ポイント

- **RAG対応**: `core_modules/rag_integration.py` を追加
- **MCP拡張**: `agentic_loop.py` でツール定義を拡充
- **マルチプロバイダ**: `providers_facade.py` でプロバイダ管理を強化

---

## 関連ドキュメント

- Issue #103: core.pyリファクタリング
- `doc/core_split_plan.md`: 実装計画
- `doc/core_split_dependencies.md`: 依存関係分析
- `doc/architecture.md`: アーキテクチャ設計（更新済み）
