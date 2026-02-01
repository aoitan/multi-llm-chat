# ドキュメント案内

プロジェクトの設計・仕様ドキュメントの置き場所と役割をまとめています（2026-02-01 現在）。

## 基本ドキュメント
- `architecture.md` — 現行3層アーキテクチャの概要。
- `development_workflow.md` — 開発フローとPR運用。
- `glossary.md` — 用語集。
- `ci_lint_spec.md` — CI/Lint 方針。

## 現行仕様 (`specs/`)
- 統合サマリ: `history_features.md`, `system_prompt_and_models.md`, `webui_design.md`
- 履歴関連: `history_feature_requirements.md`, `chat_history_reset_spec.md`, `chat_history_export_feature_spec.md`, `webui_history_management_spec.md`
- システムプロンプト/モデル: `system_prompt_feature_requirements.md`, `provider_specific_system_prompt_spec.md`, `llm_switching_feature_spec.md`
- 構造化コンテンツ: `structured_content_format_spec.md`, `migration_plan.md`
- WebUI設計: `webui_refactoring_design.md`, `webui_2pane_design_spec.md`, `gradio_ui_design.md`
- その他UI/機能: `response_copy_feature_spec.md`

## リファクタリング記録 (`refactoring/`)
- core分割: `core_split_plan.md`, `core_split_dependencies.md`, `core_split_metrics.md`
- Issue #103関連: `refactoring_dependencies.md`, `refactoring_metrics.md`

## バックログ (`backlog/`)
- `context_compression_requirements.md` — コンテキスト圧縮要件（検討中）。
- `llm_hands_up_spec.md` — 挙手制・自律応答判定案。

## アーカイブ (`archive/`)
- `spec_v0.1.md` — 初期CLI仕様。
- `development_ideas.md` — 初期アイデアメモ。

### 運用ルールのメモ
- 新規仕様は `specs/` に配置し、完了した計画や旧案は `archive/` へ移動。
- 実験中・未着手のアイデアは `backlog/` に置く。
- 大規模リファクタリングの計画・分析・結果は `refactoring/` で三点セット（Plan/Dependencies/Metrics）として管理。
