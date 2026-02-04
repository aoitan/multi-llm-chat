---
name: code-review
description: 指定されたIssueまたは変更に対して、Multi-LLMによるコードレビューを実行する。
---

# 手順

ユーザーの要求から以下のパラメータを抽出し、レビューコマンドを実行してください。

- **Issue Number**: 指定があれば `-i` で渡す。文脈から明らかな場合も補完する。
- **Base Branch**: 比較元ブランチ。指定がなければ省略（デフォルトに任せる）するか、`-b` で指定する。
- **Spec**: 追加の仕様や指示があれば、引数の最後にテキストとして渡す。

## 実行コマンド

```bash
# PATHを通す
uv tool update-shell
# レビューコマンド
llm-review -i <issue_number> -b <base_branch> "<spec text>"
```
# 制約事項

*   レビュアーの指名がない場合は --reviewers オプションを省略すること。
*   ユーザーが「すべてのレビュアー」と言及した場合のみ --reviewers all を付与する。
