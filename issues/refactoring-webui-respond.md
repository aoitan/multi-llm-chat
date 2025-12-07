# 将来のリファクタリング課題: WebUI respond()関数の責務分離

## 概要
`src/multi_llm_chat/webui.py`の`respond()`関数が複数の責務を持っており、単一責任の原則に違反している。

## 問題点（Gemini指摘 - 2025-12-02）

### 1. 入力検証とチャット機能の混在
**現状**:
```python
def respond(user_message, display_history, logic_history, system_prompt, user_id):
    # Validate user_id before processing
    if not user_id or not user_id.strip():
        display_history.append([user_message, "[System: ユーザーIDを入力してください]"])
        yield display_history, display_history, logic_history
        return
    
    # ... チャット応答処理 ...
```

**問題**: 入力検証ロジックがコアなチャット機能と混在

**推奨**: 入力検証はUIイベントハンドラ側で完結させる
```python
def validate_and_respond(user_message, ..., user_id):
    if not user_id or not user_id.strip():
        return error_response(...)
    return respond(user_message, ..., user_id)
```

### 2. UI状態管理の分散
**現状**: ボタンの有効/無効ロジックが複数の関数に分散
- `check_send_button_with_user_id()`
- `check_history_buttons_enabled()`
- `update_buttons_on_user_id()`

**推奨**: 状態管理を一元化（例: UIStateManagerクラス）

## 優先度
🟡 Medium - MVP段階では許容、将来の拡張性のため改善推奨

## 対応時期
- Epic完了後のリファクタリングフェーズ
- または、UI状態管理が複雑化したタイミング

## 関連
- Issue #29 (Story 017-A)
- Geminiレビュー（2025-12-02）
