# LLMモデル切り替え機能 仕様書

## 1. 概要

ユーザーが複数のLLMプロバイダー（Google Gemini, OpenAI, Ollama）および、それぞれのプロバイダーが提供する特定のモデルを、UIとCLIから動的に切り替えられるようにする機能。

## 2. 背景

- **柔軟性の向上**: タスクの内容（コーディング、翻訳、要約など）に応じて最適なLLMをユーザーが選択できるようにする。
- **コスト管理**: パフォーマンスとコストのバランスを考慮し、モデルを使い分けることを可能にする。
- **ローカルLLMの活用**: Ollamaをサポートすることで、オフライン環境や、より高いプライバシーが求められる場面でローカルモデルを利用できるようにする。

## 3. 機能要件

- **プロバイダー選択**: UIとCLIからLLMプロバイダー（Gemini, OpenAI, Ollama）を選択できること。
- **モデル選択**: 選択したプロバイダーで利用可能なモデルを選択できること。
- **Ollamaエンドポイント設定**: Ollamaプロバイダー利用時に、カスタムエンドポイントURLを指定できること。
- **設定の反映**: 選択された設定が即座にチャットセッションに反映されること。

## 4. UI仕様 (Gradio)

### 4.1. コンポーネント

チャットインターフェースのサイドバー、またはアコーディオンメニュー内に以下のコンポーネントを配置する。

1.  **プロバイダー選択 (Provider Selection)**
    - **UIコンポーネント**: `gr.Dropdown`
    - **ラベル**: "LLM Provider"
    - **選択肢**: `Gemini`, `OpenAI`, `Ollama`

2.  **モデル選択 (Model Selection)**
    - **UIコンポーネント**: `gr.Dropdown`
    - **ラベル**: "Model"
    - **動作**: 「プロバイダー選択」の選択内容に連動し、利用可能なモデルのリストを動的に更新する。

3.  **Ollama エンドポイント (Ollama Endpoint)**
    - **UIコンポーネント**: `gr.Textbox`
    - **ラベル**: "Ollama Endpoint URL"
    - **プレースホルダー**: `http://localhost:11434`
    - **表示条件**: 「プロバイダー選択」で `Ollama` が選択された場合のみ表示される。

### 4.2. 状態管理

- 選択されたプロバイダー、モデル、エンドポイントURLは `gr.State` を用いてセッション内で保持する。

## 5. CLI仕様 (REPL)

REPLセッション内で、プレフィックス `/` を付けたサブコマンドでモデル設定を管理する。

1.  **モデルの設定・切り替え: `/model`**
    - **書式**: `/model <provider> [model_name]`
    - **説明**: 指定したプロバイダーとモデルに切り替える。
    - **例**:
      ```
      > /model gemini 1.5-pro-latest
      Model set to: Gemini (1.5-pro-latest)
      > /model openai gpt-4o
      Model set to: OpenAI (gpt-4o)
      ```

2.  **Ollamaモデルの設定: `/model ollama`**
    - **書式**: `/model ollama <model_name> --endpoint <url>`
    - **説明**: OllamaのモデルとエンドポイントURLを設定する。エンドポイントは一度設定するとセッション内で記憶される。
    - **例**:
      ```
      > /model ollama llama3 --endpoint http://192.168.1.100:11434
      Model set to: Ollama (llama3) at http://192.168.1.100:11434
      > /model ollama codegemma
      Model set to: Ollama (codegemma) at http://192.168.1.100:11434
      ```

3.  **利用可能なモデルの表示: `/model list`**
    - **書式**: `/model list [provider]`
    - **説明**: 利用可能なモデルをプロバイダーごとに一覧表示する。
    - **例**:
      ```
      > /model list openai
      Available OpenAI Models:
      - gpt-4o
      - gpt-4-turbo
      - gpt-3.5-turbo
      ```

4.  **現在の設定表示: `/model show`**
    - **書式**: `/model show`
    - **説明**: 現在のセッションで設定されているモデル情報を表示する。
    - **例**:
      ```
      > /model show
      Current Model:
        Provider: Ollama
        Model: llama3
        Endpoint: http://192.168.1.100:11434
      ```

5.  **ヘルプ表示: `/help`**
    - **書式**: `/help`
    - **説明**: 利用可能なREPLコマンドの一覧と使い方を表示する。
