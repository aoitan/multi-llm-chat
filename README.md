# Multi-LLM Chat

## 概要
このツールは、GeminiとChatGPTの2つの大規模言語モデル（LLM）に対し、単一のインターフェースから指示を出し、会話文脈を共有しながら対話を行うためのものです。ユーザーは「司会者」として、メンション機能を用いて応答するAIを指定できます。

Web UI版とCLI版の2つのインターフェースを提供します。

## 機能
- **Web UI**: Gradioによる、チャット形式の使いやすいグラフィカルインターフェース。
- **CLI (REPL)**: 従来のコマンドラインによる対話型インターフェース。
- **統一会話履歴管理**: セッション中の全てのやり取りを時系列で保持します。
- **メンション機能**: `@gemini`, `@chatgpt`, `@all` のメンションにより、特定のLLMまたは両方のLLMに応答をルーティングします。メンションなしの入力は思考メモとして履歴にのみ追加されます。
- **API連携**: Google Gemini APIおよびOpenAI ChatGPT APIとの連携をサポートします。
- **環境変数からのAPIキー読み込み**: APIキーは環境変数または`.env`ファイルから安全に読み込まれます。

## インストール

1. **リポジトリのクローン**:
   ```bash
   git clone <リポジトリのURL>
   cd multi-llm-chat/repo
   ```

2. **`uv`のインストール**:
   Pythonのパッケージ管理ツール`uv`をインストールします。詳細な手順は[uvの公式ドキュメント](https://docs.astral.sh/uv/installation/)を参照してください。
   例 (macOS/Linux):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **仮想環境の作成とアクティベート**:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

4. **依存関係のインストール**:
   ```bash
   uv pip install google-generativeai openai python-dotenv gradio
   ```

## 使い方

### 1. APIキーの設定

プロジェクトのルートディレクトリに`.env`ファイルを作成し、以下の形式でAPIキーを設定します。
```
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```
`GOOGLE_API_KEY`は必須です。`OPENAI_API_KEY`はChatGPTを使用する場合に設定してください。

### 2. Web UI版の実行 (推奨)

仮想環境がアクティベートされていることを確認し、`app.py`を実行します。
```bash
python app.py
```
ブラウザで `http://127.0.0.1:7860` を開いてください。

#### ローカルネットワークでの共有

同じネットワーク上のスマートフォンなど、他のデバイスからアクセスしたい場合は、以下のコマンドでアプリケーションを起動します。

```bash
MLC_SERVER_NAME=0.0.0.0 python app.py
```

### 3. CLI版の実行

コマンドラインで対話を行いたい場合は、`chat_logic.py`を実行します。
```bash
python chat_logic.py
```
プロンプト(`> `)が表示されたら、以下の形式で入力します。
- **Geminiに話しかける**: `@gemini こんにちは`
- **ChatGPTに話しかける**: `@chatgpt 自己紹介して`
- **両方に話しかける**: `@all 今日の天気は？`
- **思考メモ（API呼び出しなし）**: `これはメモです`
- **終了**: `exit` または `quit`

## 開発ロードマップ
- [x] 実際のAPI呼び出しの実装
- [x] エラーハンドリングの強化
- [x] ストリーミング応答のサポート
- [x] GradioによるWeb UIの実装
- [ ] 設定ファイルの外部化
- [x] テストの追加

## ライセンス
[LICENSE](LICENSE) (TBD)