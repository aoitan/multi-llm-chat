# Provider抽象化移行の残タスク

## 完了した作業
- ✅ `LLMProvider`抽象基底クラスを作成
- ✅ `GeminiProvider`と`ChatGPTProvider`を実装
- ✅ `get_provider()`ファクトリー関数を実装
- ✅ `ChatService`を新しいプロバイダー抽象化に移行
- ✅ プロバイダーとサービスの23件のテストがパス
- ✅ コア機能の53件のテストがパス

## 残作業（優先度順）

### 1. 統合テストのモック戦略更新（高優先度）
**影響するテスト**:
- `tests/test_cli.py`: 9件のテストが失敗
- `tests/test_chat_logic.py`: 2件のテストが失敗  
- `tests/test_webui.py`: 1件のテストが失敗

**問題**:
これらのテストは`chat_logic.call_gemini_api`や`chat_logic.call_chatgpt_api`を直接モックしているが、
`ChatService`は現在`get_provider()`を使用しているため、モックが機能しない。

**解決策**:
- Option A: テストを`get_provider()`をモックするように更新
- Option B: 統合テストとして実際のAPIコールフローをテスト（モックを最小化）
- Option C: テスト用のFakeProviderを作成してインジェクション

**推奨**: Option Aを採用し、`@patch("multi_llm_chat.llm_provider.get_provider")`を使用

### 2. `core.py`のリファクタリング（中優先度）
**現状**: `call_gemini_api()`と`call_chatgpt_api()`が`core.py`に残存している

**推奨**:
- これらの関数を`GeminiProvider`と`ChatGPTProvider`のラッパーとして再実装
- または、deprecation warningを追加して将来的な削除を予告

### 3. 新規プロバイダー追加のドキュメント（低優先度）
**必要な作業**:
- `doc/architecture.md`にプロバイダー抽象化の説明を追加
- 新規プロバイダー追加のガイドを作成（例: Claude, LLaMAなど）
- Issue #55のテンプレートとして活用

## テスト実行結果（参考）
```
✅ tests/test_llm_provider.py: 9/9 passed
✅ tests/test_chat_service.py: 14/14 passed  
✅ tests/test_core.py: 30/30 passed
❌ tests/test_cli.py: 0/9 passed (モック戦略要更新)
❌ tests/test_chat_logic.py: 1/3 passed
❌ tests/test_webui.py: 20/21 passed

合計: 74/86 passed (86%)
```

## 次のステップ
1. 統合テストのモック戦略を更新（`test_cli.py`, `test_chat_logic.py`）
2. すべてのテストがパスすることを確認
3. PRを作成してレビューを依頼
