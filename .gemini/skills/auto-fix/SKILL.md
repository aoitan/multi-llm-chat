---
name: auto-fix
description: Multi-LLMループを使用して、コードの自動修正（Auto Fix）を実行する。
---

# 手順

1. ユーザーが使用したいモデル（Fixer）を指定しているか確認する。
   - 指定がない場合のデフォルトは `gemini3pro` とする。
   - 選択肢: `gemini3pro`, `gemini2.5pro`, `copilot`, `codex`, `gemini3flash`, `codex-mini`
2. ターゲットとなる Issue 番号やブランチを確認する。
3. 以下の形式でコマンドを構築・実行する。

## 実行コマンド構文

```bash
# PATHを通す
uv tool update-shell
# 注意: レビューオプションは '--' の後ろに記述する必要があります
llm-fix --fixer <model_name> -- -i <issue_number> -b <base_branch> "<spec text>"
```
