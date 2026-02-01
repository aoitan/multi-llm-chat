# 会話履歴機能 統合仕様（サマリ）

本ドキュメントは、履歴保存・管理に関する複数仕様を俯瞰し、参照先を一本化するサマリです。詳細要件は各仕様書を参照してください。

## 範囲と参照元
- 永続化・管理: `history_feature_requirements.md`
- リセット: `chat_history_reset_spec.md`
- エクスポート: `chat_history_export_feature_spec.md`
- WebUI管理: `webui_history_management_spec.md`

## ゴール
- CLI / WebUI 双方で、同等の履歴操作（保存・読み込み・リセット・エクスポート）を提供する。
- 将来の認証導入や複数ユーザー運用を見据え、IDスキームとストレージを拡張可能に保つ。

## 機能セット
- 保存・読み込み（複数履歴 / ユーザーID切替）
- 履歴クリア（セッションリセット）
- エクスポート（テキスト / JSON）
- WebUIでの一覧・読み込み・新規開始

## UIマッピング
- CLI: `history_feature_requirements.md` に定義されたコマンド群。
- WebUI: `webui_history_management_spec.md` に定義された一覧・ロード・新規開始フロー。リセット／エクスポートはボタンで提供。

## 開発メモ
- 仕様間の重複は本サマリを入口とし、詳細は各ファイルに分離維持。
- ストレージとユーザーIDポリシーは将来の認証導入時に再検討すること。
