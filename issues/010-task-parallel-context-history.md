# Task: フェーズ2（並行作業可能）— コンテキスト圧縮と会話履歴管理

## 概要
フェーズ1完了後に着手できる並行タスク群を1ファイルずつ割り当てられるように再編したタスク票。  
`doc/context_compression_requirements.md` と `doc/history_feature_requirements.md` をそれぞれ実装対象とし、両者は同じ基盤（システムプロンプト済み`core.py`）を共有しつつ互いに依存しない。

## 並行実行の前提
- フェーズ1で`core.py`・システムプロンプト・`get_token_info`の共通インターフェースが揃っていること。
- トークン数・履歴フォーマットの仕様が確定し、双方の作業で変更が競合しない状態であること。

## 作業項目（担当割り当て用に2軸で整理）
1. **Task A: コンテキスト圧縮 / トークンガードレール**
   - モデル別最大コンテキスト長の環境変数ローダとフォールバック。
   - プロバイダ別トークン計測（OpenAI=tiktoken、Gemini=概算+バッファ）と`TOKEN_ESTIMATION_BUFFER_FACTOR`。
   - スライディングウィンドウ枝刈りをGemini/OpenAI送信フローへ組み込み、システムプロンプト保持を保証。
   - 超過時のUI警告・CLIエラー出力、ログ記録。

2. **Task B: 会話履歴の保存・読み込み**
   - ユーザーID入力と警告表示（Web UI / CLI）。
   - `chat_histories/<user>/<name>.json`フォーマットのI/O、サニタイズ、`.gitignore`設定。
   - Web UIの履歴パネルとCLI `/history`コマンド群（list/save/load/new）。
   - 未保存破棄や上書き確認のUX整備。

## 完了条件
- Task A/Bの成果物を独立にレビュー・マージでき、互いの作業を待たずに検証可能。
- README/AGENTS/`.env.example`に新環境変数や操作手順が追記されている。
- `pytest`が両タスクのテスト（トークン計算・履歴操作）を網羅し、CIがグリーンである。
