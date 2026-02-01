# core.py 分割案の評価と実装計画

## 📋 提案された分割案（要約）

```
src/multi_llm_chat/
├── core.py                    # 薄いファサード（再エクスポートのみ）
└── core/
    ├── __init__.py            # パッケージ初期化
    ├── legacy_api.py          # DEPRECATED wrapper群
    ├── token_and_context.py   # トークン計算と検証
    ├── agentic_loop.py        # Agentic Loop実装
    └── providers_facade.py    # Provider関連の入口
```

---

## ✅ 評価：非常に優れた提案

### 強み

1. **責務の明確化**
   - 各モジュールが単一の責務を持つ
   - DEPRECATED層の明確な隔離

2. **段階的な廃止が容易**
   - `legacy_api.py` を将来的に削除しやすい
   - 他のモジュールへの影響を最小化

3. **テストの整理が自然**
   - モジュール構造とテスト構造が対応
   - 既存の `test_agentic_loop*.py` との統合が容易

4. **拡張性**
   - `agentic_loop.py` に RAG/MCP 拡張を集約
   - `providers_facade.py` で新しい Provider の追加が容易

---

## 🔍 現状分析

### core.py の内訳（740行）

| カテゴリ | 関数/クラス | 行数（概算） | 提案された移動先 |
|---------|------------|------------|----------------|
| **Agentic Loop** | `AgenticLoopResult`, `execute_with_tools_*` (3関数) | ~410行 | `core/agentic_loop.py` |
| **DEPRECATED wrapper** | `call_*_api*`, `stream_text_events*`, `extract_text_from_chunk`, `format_history_for_*`, `load_api_key` | ~95行 | `core/legacy_api.py` |
| **トークン・検証** | `calculate_tokens`, `get_token_info`, `prune_*`, `validate_*`, `get_max_context_length`, `_estimate_tokens` | ~70行 | `core/token_and_context.py` |
| **Provider関連** | `list_gemini_models` | ~27行 | `core/providers_facade.py` |
| **Import・定数** | - | ~138行 | `core.py` (ファサード) |

### テストの内訳（992行）

| ファイル | 行数 | 対応する分割先 |
|---------|------|---------------|
| `test_core.py` | 421行 | 分割対象 |
| `test_agentic_loop.py` | 331行 | `test_agentic_loop.py` (統合) |
| `test_agentic_loop_immutability.py` | 240行 | `test_agentic_loop.py` (統合) |

---

## 📐 詳細な分割計画

### 1️⃣ `core/legacy_api.py` (~140行)

**移動対象**:
```python
# DEPRECATED API Wrappers
def call_gemini_api_async(history, system_prompt=None)       # ~16行
def call_gemini_api(history, system_prompt=None)             # ~27行
def call_chatgpt_api_async(history, system_prompt=None)      # ~16行
def call_chatgpt_api(history, system_prompt=None)            # ~27行
def stream_text_events_async(history, provider_name, ...)    # ~8行
def stream_text_events(history, provider_name, ...)          # ~9行
def extract_text_from_chunk(chunk, model_name)               # ~15行
def format_history_for_gemini(history)                       # ~7行
def format_history_for_chatgpt(history)                      # ~7行
def load_api_key(env_var_name)                               # ~9行
def prepare_request(history, system_prompt, model_name)      # ~3行
```

**依存関係**:
- `llm_provider` (Provider層)
- `history_utils` (prepare_request)

**テスト**: `tests/test_core_legacy_api.py` (~150行)

---

### 2️⃣ `core/token_and_context.py` (~110行)

**移動対象**:
```python
# Token calculation
def _estimate_tokens(text)                                   # ~2行 (wrapper)
def calculate_tokens(text: str, model_name: str) -> int      # ~8行
def get_token_info(text, model_name, history=None)          # ~26行
def get_max_context_length(model_name)                       # ~2行 (wrapper)

# History pruning
def prune_history_sliding_window(history, max_tokens, ...)  # ~6行 (wrapper)
def get_pruning_info(history, max_tokens, ...)              # ~6行 (wrapper)

# Validation
def validate_system_prompt_length(system_prompt, model_name) # ~6行 (wrapper)
def validate_context_length(history, system_prompt, ...)     # ~6行 (wrapper)
```

**依存関係**:
- `token_utils` (実装委譲先)
- `compression` (実装委譲先)
- `validation` (実装委譲先)
- `llm_provider` (Provider取得)
- `history_utils` (get_provider_name_from_model)

**注意**: これらは既に委譲済みなので、実質的には **wrapper の集約** となる。

**テスト**: `tests/test_token_and_context.py` (~120行)

---

### 3️⃣ `core/agentic_loop.py` (~450行)

**移動対象**:
```python
# Data structure
@dataclass(frozen=True)
class AgenticLoopResult                                      # ~21行

# Agentic Loop implementation
async def execute_with_tools_stream(provider, history, ...) # ~169行
async def execute_with_tools(provider, history, ...)        # ~144行
def execute_with_tools_sync(provider, history, ...)         # ~53行
```

**依存関係**:
- `llm_provider` (Provider抽象化)
- `mcp.client` (MCPClient)
- `history_utils` (validate_history_entry)
- `asyncio`, `logging`

**テスト**: 
- `tests/test_agentic_loop.py` (既存 331行 + 移動分 ~80行 = **410行**)
- `tests/test_agentic_loop_immutability.py` (既存 240行) → 統合または独立

---

### 4️⃣ `core/providers_facade.py` (~80行)

**移動対象**:
```python
# Provider factory (実際は llm_provider から re-export)
# create_provider, get_provider は llm_provider.py に既に存在

# Debug utility
def list_gemini_models(verbose: bool = True)                 # ~27行

# Helper (実際は history_utils から re-export)
# get_provider_name_from_model は history_utils.py に既に存在
```

**注意**: このモジュールは実質的に以下を行う：
1. `list_gemini_models()` の実装を保持
2. `llm_provider` からの re-export を集約
3. Provider関連の入口として機能

**テスト**: `tests/test_provider_access.py` (~50行)

---

### 5️⃣ `core.py` (ファサード、~150行)

**役割**: 
- 上記4モジュールからの公開APIを re-export
- 環境変数・定数の re-export (`GOOGLE_API_KEY`, `GEMINI_MODEL`, `MCP_ENABLED` 等)
- Backward compatibility の維持

**構成**:
```python
"""Core module - Facade for multi_llm_chat

This module provides a unified interface to various sub-modules.
For new code, consider importing directly from sub-modules:
- core.agentic_loop: Agentic Loop execution
- core.token_and_context: Token calculation and validation
- core.providers_facade: Provider management
- core.legacy_api: DEPRECATED wrapper functions
"""

# Re-export from sub-modules
from .core.agentic_loop import (
    AgenticLoopResult,
    execute_with_tools,
    execute_with_tools_stream,
    execute_with_tools_sync,
)
from .core.legacy_api import (
    call_chatgpt_api,
    call_gemini_api,
    extract_text_from_chunk,
    format_history_for_chatgpt,
    format_history_for_gemini,
    load_api_key,
    stream_text_events,
    # ... 他のDEPRECATED API
)
from .core.providers_facade import (
    list_gemini_models,
)
from .core.token_and_context import (
    calculate_tokens,
    get_max_context_length,
    get_pruning_info,
    get_token_info,
    prune_history_sliding_window,
    validate_context_length,
    validate_system_prompt_length,
)

# Re-export from llm_provider
from .llm_provider import (
    CHATGPT_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    MCP_ENABLED,
    OPENAI_API_KEY,
)

__all__ = [
    # Agentic Loop
    "AgenticLoopResult",
    "execute_with_tools",
    # ... 全公開API
]
```

---

## 🚦 段階的実装計画（10コミット）

### Phase 1: 準備（2コミット、各 < 50行）

#### Commit 1: core/ パッケージの作成
- `src/multi_llm_chat/core/__init__.py` 作成（空ファイル）
- パッケージ構造の確認

#### Commit 2: 依存関係の詳細分析
- `doc/core_split_dependencies.md` 作成
- 各関数の import 依存を詳細にマッピング

---

### Phase 2: 関数移動（4コミット、各 < 150行）

#### Commit 3: `core/legacy_api.py` 作成
- DEPRECATED wrapper 群を移動
- `core.py` からは re-export
- `tests/test_core_legacy_api.py` 作成（`test_core.py` から該当テスト移動）

#### Commit 4: `core/token_and_context.py` 作成
- トークン計算・検証 wrapper を移動
- `core.py` からは re-export
- `tests/test_token_and_context.py` 作成（`test_core.py` から該当テスト移動）

#### Commit 5: `core/agentic_loop.py` 作成
- Agentic Loop 実装を移動
- `core.py` からは re-export
- 既存 `test_agentic_loop*.py` に追加テストを統合

#### Commit 6: `core/providers_facade.py` 作成
- `list_gemini_models()` を移動
- Provider関連の re-export を集約
- `tests/test_provider_access.py` 作成

---

### Phase 3: クリーンアップ（3コミット、各 < 100行）

#### Commit 7: `test_core.py` の最終整理
- 移動後に残ったテストを確認
- ファサード経由の統合テストのみ残す（最小限）

#### Commit 8: `core.py` をファサードに縮小
- すべての実装を削除
- re-export のみに変更
- import の最適化

#### Commit 9: ドキュメント更新
- `doc/architecture.md` 更新（core/ パッケージの説明）
- `README.md` 更新（モジュール構成図）
- `doc/core_split_metrics.md` 作成（分割結果の記録）

---

### Phase 4: 検証（1コミット）

#### Commit 10: 最終検証とメトリクス
- 全テスト実行（283件全パス確認）
- カバレッジ確認
- コミットログの整理

---

## 📊 期待される効果

### Before (現状)

| ファイル | 行数 | 役割 |
|---------|------|------|
| `core.py` | 740行 | すべて混在 |
| `test_core.py` | 427行, 28テスト | すべて混在（単一ファイル） |

### After (分割後)

| ファイル | 行数 | 役割 |
|---------|------|------|
| `core.py` | 129行 | ファサード（re-export） ✅ |
| `core_modules/legacy_api.py` | 290行 | DEPRECATED wrapper ✅ |
| `core_modules/token_and_context.py` | 201行 | トークン・検証 wrapper ✅ |
| `core_modules/agentic_loop.py` | 423行 | Agentic Loop実装 ✅ |
| `core_modules/providers_facade.py` | 45行 | Provider入口 ✅ |
| **合計** | **1,101行** | （import増加分含む） |

| テストファイル | 行数 | 役割 |
|---------------|------|------|
| `tests/test_core_token_context.py` | 90行, 6テスト | トークン・Context管理 ✅ |
| `tests/test_core_legacy_api.py` | 280行, 20テスト | DEPRECATED API ✅ |
| `tests/test_core_facade.py` | 50行, 2テスト | ファサード検証 ✅ |
| **合計** | **420行, 28テスト** | （機能別に分割、テスト総数維持） |

**実装結果**: 
- ✅ core.py は129行まで削減（82%減）
- ✅ テストは機能別に3ファイルへ分割（28テスト維持）
- ✅ 全285テストが通過（破壊的変更なし）

---

## ⚠️ リスクと対策

### リスク1: Import循環依存
**対策**: 各モジュールは他の `core/*` を import せず、`llm_provider`, `history_utils` 等の既存モジュールのみに依存

### リスク2: 既存コードへの影響
**対策**: 
- `core.py` からの re-export により完全な互換性を維持
- 段階的な移行で各コミット後にテスト実行

### リスク3: テストの重複
**対策**: 
- wrapper テストは最小限に（公開API互換性のみ）
- 実装テストはサブモジュールで完結

---

## 🎯 完了ステータス

**Issue #103 完了条件**:
1. ✅ **core.pyの行数・責務が縮小** → 740行 → 129行（82%削減）
2. ✅ **テストが機能別ファイルに分割** → `test_core.py`（427行）を3ファイル（420行）に分割
3. ✅ **pytestが通る** → 全285テスト通過

**完了日時**: 2026-02-01  
**実装フェーズ**:
- Phase 1: 準備（2コミット）
- Phase 2: モジュール分割（4コミット）
- Phase 3: テスト修正（3コミット）
- Phase 4: テスト分割（2コミット）
- Phase 5: 検証と文書化（1コミット）

**成果**: Issue #103の全要件を満たし、保守性と可読性が向上しました。
