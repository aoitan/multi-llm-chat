# core.py 分割: 詳細な依存関係分析

本ドキュメントは core.py を core_modules/ パッケージに分割する際の、
各関数の依存関係を詳細に記録したものです。

## Import 依存関係の概要

core.py は以下のモジュールに依存しています：

### 標準ライブラリ
- `asyncio`: 非同期処理
- `logging`: ロギング
- `dataclasses`: AgenticLoopResult データクラス
- `typing`: 型ヒント

### サードパーティ
- `google.generativeai as genai`: list_gemini_models() で使用
- `openai`: （インポートのみ、直接使用なし）
- `dotenv`: 環境変数読み込み

### プロジェクト内モジュール
- `.compression`: prune_history_sliding_window, get_pruning_info
- `.history_utils`: LLM_ROLES, get_provider_name_from_model, prepare_request, validate_history_entry
- `.llm_provider`: 定数（MODEL, API_KEY等）、Provider クラス、create_provider, get_provider
- `.token_utils`: estimate_tokens, get_max_context_length
- `.validation`: validate_system_prompt_length, validate_context_length

---

## 分割モジュールごとの依存関係

### 1. `core_modules/legacy_api.py`

**移動対象の関数**:
- `load_api_key(env_var_name)`
- `format_history_for_gemini(history)`
- `format_history_for_chatgpt(history)`
- `extract_text_from_chunk(chunk, model_name)`
- `prepare_request(history, system_prompt, model_name)`
- `call_gemini_api_async(history, system_prompt=None)`
- `call_gemini_api(history, system_prompt=None)`
- `call_chatgpt_api_async(history, system_prompt=None)`
- `call_chatgpt_api(history, system_prompt=None)`
- `stream_text_events_async(history, provider_name, system_prompt=None)`
- `stream_text_events(history, provider_name, system_prompt=None)`

**必要な import**:
```python
import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..llm_provider import (
    GeminiProvider,
    ChatGPTProvider,
    create_provider,
    get_provider,
    GOOGLE_API_KEY,
)
from ..history_utils import (
    get_provider_name_from_model,
    prepare_request as _prepare_request,
)
```

**依存先**:
- `llm_provider`: Provider層への委譲
- `history_utils`: get_provider_name_from_model, prepare_request

**内部依存**: なし（他の core_modules を import しない）

---

### 2. `core_modules/token_and_context.py`

**移動対象の関数**:
- `_estimate_tokens(text)`
- `get_max_context_length(model_name)`
- `calculate_tokens(text: str, model_name: str) -> int`
- `get_token_info(text, model_name, history=None) -> Dict[str, Any]`
- `prune_history_sliding_window(history, max_tokens, model_name, system_prompt=None)`
- `get_pruning_info(history, max_tokens, model_name, system_prompt=None)`
- `validate_system_prompt_length(system_prompt, model_name)`
- `validate_context_length(history, system_prompt, model_name)`

**必要な import**:
```python
from typing import Any, Callable, Dict, List, Optional

from ..compression import (
    get_pruning_info as _get_pruning_info,
    prune_history_sliding_window as _prune_history_sliding_window,
)
from ..history_utils import get_provider_name_from_model
from ..llm_provider import get_provider, TIKTOKEN_AVAILABLE
from ..token_utils import (
    estimate_tokens as _estimate_tokens_impl,
    get_max_context_length as _get_max_context_length,
)
from ..validation import (
    validate_context_length as _validate_context_length,
    validate_system_prompt_length as _validate_system_prompt_length,
)
```

**依存先**:
- `compression`: prune/pruning_info の実装
- `validation`: validate_* の実装
- `token_utils`: estimate_tokens, get_max_context_length の実装
- `llm_provider`: Provider取得、TIKTOKEN_AVAILABLE
- `history_utils`: get_provider_name_from_model

**内部依存**: なし

---

### 3. `core_modules/agentic_loop.py`

**移動対象の関数/クラス**:
- `AgenticLoopResult` (dataclass)
- `execute_with_tools_stream(provider, history, system_prompt, mcp_client, ...)`
- `execute_with_tools(provider, history, system_prompt, mcp_client, ...)`
- `execute_with_tools_sync(provider, history, system_prompt, mcp_client, ...)`

**必要な import**:
```python
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..history_utils import validate_history_entry
from ..llm_provider import LLMProvider  # 型ヒント用
from ..mcp.client import MCPClient  # 型ヒント用
```

**依存先**:
- `history_utils`: validate_history_entry
- `llm_provider`: LLMProvider (型ヒント)
- `mcp.client`: MCPClient (型ヒント)

**内部依存**: なし

**注意**: 
- `execute_with_tools_stream` 内で `provider.call_api()` を呼び出すが、
  これは引数として受け取った provider オブジェクトのメソッドなので、
  llm_provider モジュールへの実行時依存はない（型ヒントのみ）

---

### 4. `core_modules/providers_facade.py`

**移動対象の関数**:
- `list_gemini_models(verbose: bool = True)`

**再エクスポート対象** (llm_provider から):
- `create_provider`
- `get_provider`

**再エクスポート対象** (history_utils から):
- `get_provider_name_from_model`

**必要な import**:
```python
import logging
from typing import List

import google.generativeai as genai

from ..llm_provider import (
    GOOGLE_API_KEY,
    create_provider,
    get_provider,
)
from ..history_utils import get_provider_name_from_model

logger = logging.getLogger(__name__)
```

**依存先**:
- `google.generativeai`: list_gemini_models 実装
- `llm_provider`: GOOGLE_API_KEY, create_provider, get_provider
- `history_utils`: get_provider_name_from_model

**内部依存**: なし

---

## Import 順序の注意点

### 1. dotenv の読み込みタイミング

現在 core.py では以下の順序でimportしています：

```python
import asyncio
import logging
# ...
from dotenv import load_dotenv

load_dotenv()  # ← 環境変数読み込み

# その後にプロジェクト内モジュールをimport
from .llm_provider import GOOGLE_API_KEY
```

**対応**: 
- 各 core_modules/*.py では `dotenv.load_dotenv()` を呼ばない
- 既に llm_provider.py で load_dotenv() が実行されているため問題なし

### 2. 循環import の回避

**ルール**: 
- 各 `core_modules/*.py` は **他の core_modules をimportしない**
- すべて既存モジュール（llm_provider, history_utils等）への依存のみ
- core.py のファサード化後、core.py から各モジュールをimportする一方向の依存

---

## 移動時のチェックリスト

各関数を移動する際、以下を確認：

### 関数本体の移動

- [ ] 関数シグネチャが完全一致
- [ ] docstring を含めてコピー
- [ ] DEPRECATED コメントがあれば維持

### Import の調整

- [ ] 必要な import のみ追加（未使用import回避）
- [ ] 相対import (`..*`) を使用
- [ ] 型ヒントに必要な import も忘れずに

### テストの確認

- [ ] 移動後に pytest が全パス
- [ ] core.py 経由の呼び出しが動作（re-export確認）

---

## Phase 2 実装順序の根拠

**Commit 3: legacy_api.py → 最初**
- 他モジュールへの依存が少ない（Provider層への委譲のみ）
- テスト範囲が明確

**Commit 4: token_and_context.py → 2番目**
- wrapper 集約のため、実装は薄い
- 依存先は既存モジュールのみ

**Commit 5: agentic_loop.py → 3番目**
- 最も行数が多いが、依存関係はシンプル
- validate_history_entry のみ使用

**Commit 6: providers_facade.py → 最後**
- list_gemini_models の移動
- re-export の集約

この順序により、各コミットで独立してテスト可能。
