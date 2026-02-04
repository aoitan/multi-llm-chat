---
name: get-pr-review
description: 指定されたPR番号のレビューコメントを取得し、要約しやすい形式でJSON出力する。PRの内容把握やレビュー状況の分析時に使用する。
---

# 手順
以下のコマンドを実行して、Pull Requestのコメントを取得してください。

```bash
gh api "repos/:owner/:repo/pulls/<pr number>/comments" | jq '.[] | { login: .user.login, update_date: .updated_at, body }'
```

制約事項

*   <pr number> はユーザーの要求またはコンテキストから推測される番号に置換すること。
*   出力が長すぎる場合は直近の10件に絞ること。
