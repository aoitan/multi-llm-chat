---
name: get_pr_review
description: GithubのPRからレビューコメントを得る
---

# 手順
1. gh api "repos/:owner/:repo/pulls/<pr number>/comments" | jq '.[] | { login: .user.login, update_date: .updated_at, body }'
