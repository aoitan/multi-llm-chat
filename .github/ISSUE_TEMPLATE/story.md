---
name: 'Story'
about: 'ユーザー価値のある開発単位'
title: 'Story [Epic ID]-[Story ID]: [タイトル]'
labels: 'story'
---

## 概要 (Overview)

<!-- このStoryで提供するユーザー価値を記述してください -->

## 親Epic (Parent Epic)

- Epic: #[Epic番号] - [Epicタイトル]

## 仕様書 (Specifications)

- [関連仕様書へのリンク](../doc/[spec_file].md)

## 受け入れ基準 (Acceptance Criteria)

- [ ] 基準1
- [ ] 基準2
- [ ] 基準3

## タスク分割 (Tasks)

- [ ] #[Task番号] - Task [Epic ID]-[Story ID]-1: [タスクタイトル]
- [ ] #[Task番号] - Task [Epic ID]-[Story ID]-2: [タスクタイトル]
- [ ] #[Task番号] - Task [Epic ID]-[Story ID]-3: [タスクタイトル]

## ブランチ戦略 (Branch Strategy)

- Epic ブランチ: `epic/[epic_id]-[description]`
- Story ブランチ: `story/[epic_id]-[story_id]-[description]`
- Task ブランチ:
  - `task/[epic_id]-[story_id]-1-[description]`
  - `task/[epic_id]-[story_id]-2-[description]`

## PR フロー (PR Flow)

1. 各 Task ブランチ → Story ブランチへPR
2. Story完了後 → Epic ブランチへPR
3. Epic完了後 → `main` へPR

## その他 (Notes)

<!-- 補足事項があれば記述してください -->
