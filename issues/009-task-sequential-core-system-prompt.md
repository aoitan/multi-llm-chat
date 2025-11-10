# Task: フェーズ1（並行作業不可）— Coreリファクタリングとシステムプロンプト実装

## 概要
READMEの「実装の優先順位」で示されている、コンテキスト圧縮/履歴機能より前に着手すべき作業を切り出したタスク票。  
`doc/system_prompt_feature_requirements.md` で詳細化された計画を1つの担当範囲にまとめ、後続フェーズの前提条件を満たす。

## 並行作業不可の理由
- 既存`app.py`/`chat_logic.py`から`core.py`/`webui.py`/`cli.py`へ責務を再配置しないと、以降の機能で共通ロジックを再利用できない。
- システムプロンプト機能は、コンテキスト圧縮や履歴永続化が参照するデータ形を決めるため、実装完了まで他フェーズを開始できない。

## 作業項目
1. **Core層の新設とエントリーポイント再編**
   - `core.py`へAPIキー読み込み、モデルクライアント生成、履歴フォーマッタ、`get_token_info`スタブを移植。
   - `app.py`→`webui.py`、`chat_logic.py`→`cli.py`へのリネームとimport更新。
   - テストを`tests/test_core.py`と`tests/test_cli.py`へ分割。

2. **システムプロンプト適用ロジック**
   - `prepare_request(history, system_prompt, model_name)`で各LLMの履歴整形を統一。
   - Geminiキャッシュの再初期化、`get_token_info`の暫定実装を完了。

3. **UI/CLIインターフェース統合**
   - Gradioにシステムプロンプト編集欄とトークン残量表示を追加し、超過時は送信をブロック。
   - CLIに`/system`系コマンドを追加し、トークン上限チェックとエラーメッセージを実装。

4. **永続化フローへの組み込み**
   - 履歴保存・読み込み処理で`system_prompt`フィールドを扱う（`doc/history_feature_requirements.md`との整合性確保）。

## 完了条件
- `core.py`, `webui.py`, `cli.py`が新しい責務分割を反映し、旧`app.py`/`chat_logic.py`は廃止。
- Web UI / CLIの双方でシステムプロンプト編集・表示・検証が動作し、`get_token_info`の結果が連動。
- 履歴保存データにシステムプロンプトが含まれ、読み込みで復元される。
- 既存および新規テストが`pytest`でパスし、README/AGENTSに起動手順と新機能が反映されている。
