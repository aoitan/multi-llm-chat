# 構造化コンテンツへの移行計画

## 概要
このドキュメントは、会話履歴の内部表現を従来の文字列形式（`content: str`）から構造化形式（`content: List[Dict]`）へ段階的に移行するロードマップを定義します。

## 背景
- **従来形式**: `{"role": "user", "content": "Hello"}`
- **新形式**: `{"role": "user", "content": [{"type": "text", "content": "Hello"}]}`

新形式は以下をサポートします：
- テキスト: `{"type": "text", "content": str}`
- ツール呼び出し: `{"type": "tool_call", "content": {"name": str, "arguments": dict}}`
- ツール結果: `{"type": "tool_result", "name": str, "content": Any, "tool_call_id": str}`

## 移行フェーズ

### Phase 1: デュアルサポート期間（現在 - v1.x系）
**状態**: 実装済み（v1.5.0）

**実装内容**:
- [x] `normalize_history_turns()` による自動変換
- [x] `content_to_text()` による後方互換API
- [x] 既存テストの更新（両形式をサポート）
- [x] 新規テストの追加（`test_history_utils.py`）

**ユーザーへの影響**:
- 既存のJSONファイルは自動的に新形式に変換されて保存される
- レガシー形式のコードも引き続き動作する

---

### Phase 2: 非推奨化（6ヶ月後 - v2.0.0）
**予定**: 2026年7月リリース

**実装予定**:
- [ ] `content: str` 形式の利用時に警告ログを出力
  ```python
  warnings.warn(
      "String-based content is deprecated. Use List[Dict] format.",
      DeprecationWarning
  )
  ```
- [ ] 新規API（`ChatService.process_message`など）は構造化形式のみ受付
- [ ] ドキュメントに移行ガイドを追加

**ユーザーへの影響**:
- レガシー形式のコードは動作するが、警告が表示される
- 新規開発では構造化形式の利用が推奨される

**移行方法**（ユーザー向け）:
```python
# Before (v1.x)
history.append({"role": "user", "content": "Hello"})

# After (v2.0+)
history.append({
    "role": "user",
    "content": [{"type": "text", "content": "Hello"}]
})
```

---

### Phase 3: レガシー形式の削除（12ヶ月後 - v3.0.0）
**予定**: 2027年1月リリース

**実装予定**:
- [ ] `normalize_history_turns()` を内部関数化（公開APIから削除）
- [ ] `content_to_text(include_tool_data=False)` の削除
- [ ] レガシー形式を扱うテストケースの削除
- [ ] `HistoryStore` のスキーマバージョン1のサポート終了

**ユーザーへの影響**:
- レガシー形式のコードは動作しなくなる
- v2.x で非推奨警告を修正していれば影響なし

---

## 開発者向けガイドライン

### 新規コードを書く際の注意
1. **常に構造化形式を使用**
   ```python
   # Good
   content = [{"type": "text", "content": "Hello"}]
   
   # Bad (avoid in new code)
   content = "Hello"
   ```

2. **`content_to_text()` は一時的なAPI**
   - トークン計算や圧縮など、テキスト化が必要な場合のみ使用
   - 新規機能では直接 `content` を扱う

3. **テストコードの書き方**
   - 新規テストは構造化形式のみをテスト
   - レガシー形式のテストは Phase 2 で削除予定

### Phase 2 実装時のタスクリスト
- [ ] `history_utils.py` に `@deprecated` デコレータを追加
- [ ] `HistoryStore.load_history()` で警告ログを出力
- [ ] ドキュメント `doc/api_reference.md` を更新
- [ ] チェンジログに Breaking Changes を記載

### Phase 3 実装時のタスクリスト
- [ ] `content_to_text()` の削除（または内部API化）
- [ ] `normalize_history_turns()` を `_normalize_history_turns()` にリネーム
- [ ] スキーマバージョン1のサポートコードを削除
- [ ] 影響を受けるテストケースをすべて削除

---

## FAQ

### Q: 既存の保存済み履歴ファイルはどうなる？
A: Phase 1 で自動的に新形式に変換されて再保存されます。元のファイルは上書きされますが、動作に影響はありません。

### Q: Phase 2 の警告を無視したらどうなる？
A: Phase 3 で動作しなくなります。Phase 2 期間中（6ヶ月間）にすべての警告を修正してください。

### Q: カスタムプロバイダを実装している場合は？
A: `call_api()` が統一辞書形式（`{"type": "text"|"tool_call", "content": ...}`）を返すように修正してください。詳細は `doc/custom_provider.md` を参照。

---

## 関連ドキュメント
- [アーキテクチャドキュメント](./architecture.md)
- [開発ワークフロー](./development_workflow.md)
- [Issue #78](../issues/issue_078_function_calling_support.md) - Function Calling サポート

---

**最終更新**: 2026-01-12  
**担当**: @copilot-dev
