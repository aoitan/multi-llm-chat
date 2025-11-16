# 011: Epic - LLMモデル切り替え機能の実装

## 概要

ユーザーがGradio UIおよびCLIのREPLから、LLMプロバイダー（Gemini, OpenAI, Ollama）と、それに対応するモデルを動的に選択・切り替えられるようにする機能を実装します。

## 関連仕様書

- [LLMモデル切り替え機能 仕様書](../doc/llm_switching_feature_spec.md)

## 受け入れ基準 (Acceptance Criteria)

- **UI (Gradio)**
  - [ ] UIに「プロバイダー選択」ドロップダウンが実装されている。
  - [ ] UIに「モデル選択」ドロップダウンが実装され、プロバイダー選択に連動して内容が更新される。
  - [ ] プロバイダーで「Ollama」を選択した時のみ、「Ollamaエンドポイント」の入力欄が表示される。
  - [ ] UIで選択したモデル設定がチャットの実行に正しく反映される。
- **CLI (REPL)**
  - [ ] REPL内で `/model` サブコマンドが利用可能になっている。
  - [ ] `/model <provider> [model_name]` でモデルを切り替えられる。
  - [ ] `/model ollama <model_name> --endpoint <url>` でOllamaのエンドポイントとモデルを設定できる。
  - [ ] `/model list` で利用可能なモデルの一覧が表示される。
  - [ ] `/model show` で現在のモデル設定が表示される。
  - [ ] `/help` で新コマンドのヘルプが表示される。
- **共通**
  - [ ] 選択されたモデル（Gemini, OpenAI, Ollama）に応じて、適切なAPIクライアントが使用される。
  - [ ] 新機能に関するユニットテストが追加され、すべてのテストがパスする。

## タスク分割 (Sub-tasks)

1.  **Core**: `src/multi_llm_chat/core.py` に、複数のLLMクライアント（Gemini, OpenAI, Ollama）を管理するファクトリーやクラス構造を実装する。
2.  **Chat Logic**: `src/multi_llm_chat/chat_logic.py` を修正し、現在のモデル設定に応じて `core` の適切なクライアントを呼び出すようにする。
3.  **Web UI**: `src/multi_llm_chat/webui.py` に、仕様書通りのGradioコンポーネント（ドロップダウン、テキストボックス）と、それらを制御するロジックを実装する。
4.  **CLI**: `src/multi_llm_chat/cli.py` のREPLに、`/model` サブコマンド群を処理するパーサーと実行ロジックを実装する。
5.  **Testing**: `tests/` ディレクトリに、`test_core.py`, `test_chat_logic.py`, `test_cli.py` の変更に対応するユニットテストを追加・修正する。
6.  **Documentation**: `README.md` やその他のドキュメントに、新しいモデル切り替え機能の使い方を追記する。
