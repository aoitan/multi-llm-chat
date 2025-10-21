# Multi-LLM CLI Tool Specification

## 1. 概要

GeminiとChatGPTの2つのLLMに対し、単一のCLIインターフェースから指示を出し、会話文脈を共有しながら対話を行うためのツール。
ユーザーは「司会者」として、メンション機能を用いて応答するAIを指定する。

## 2. コア機能

### 2.1. REPL (Read-Eval-Print Loop)

- ツールを起動すると、ユーザーの入力を待ち受けるプロンプト (`> `など) が表示される。
- ユーザーの入力（プロンプト）を受け付け、評価（Eval）し、結果（Print）を標準出力に表示した後、再び入力待ちに戻る。
- ループは `exit` または `quit` コマンドで終了する。

### 2.2. 文脈（会話履歴）の管理

- ツールは、セッション中のすべてのやり取りを時系列で保持する「統一会話履歴」リストを持つ。
- このリストには以下の情報がすべて含まれる：
    - ユーザーの発言（メンションの有無、対象を問わず）
    - Geminiの回答
    - ChatGPTの回答

### 2.3. メンション機能（ルーティング）

ユーザーの入力は、先頭の文字列によって処理が分岐される。

#### A. `@gemini <prompt>`

- **トリガー:** プロンプトが `@gemini` で始まる。
- **アクション:**
    1. 現在の「統一会話履歴」全体を取得する。
    2. ユーザーの現在の発言 (`@gemini <prompt>`) を履歴に追加する。
    3. 最終化された履歴を Gemini API に送信する。
    4. Geminiからの回答 (`Response_G`) を受け取る。
    5. `Response_G` を「統一会話履歴」に追加する。
    6. `Response_G` をCLIに表示する。

#### B. `@chatgpt <prompt>`

- **トリガー:** プロンプトが `@chatgpt` で始まる。
- **アクション:**
    1. 現在の「統一会話履歴」全体を取得する。
    2. ユーザーの現在の発言 (`@chatgpt <prompt>`) を履歴に追加する。
    3. 最終化された履歴を ChatGPT API (OpenAI) に送信する。
    4. ChatGPTからの回答 (`Response_C`) を受け取る。
    5. `Response_C` を「統一会話履歴」に追加する。
    6. `Response_C` をCLIに表示する。

#### C. `@all <prompt>`

- **トリガー:** プロンプトが `@all` で始まる。
- **アクション:**
    1. 現在の「統一会話履歴」全体を取得する。
    2. ユーザーの現在の発言 (`@all <prompt>`) を履歴に追加する。
    3. **(並列または直列で) 両方のAPIを呼び出す。**
        - Gemini API に履歴を送信し、`Response_G` を受け取る。
        - ChatGPT API に履歴を送信し、`Response_C` を受け取る。
    4. `Response_G` と `Response_C` を（順序を担保して）「統一会話履歴」に追加する。
    5. `Response_G` と `Response_C` をCLIに表示する（例: `[Gemini]: ...`, `[ChatGPT]: ...`）。

#### D. `<prompt>` (メンションなし / 思考メモ)

- **トリガー:** プロンプトが `@` で始まらない（`exit`/`quit` を除く）。
- **アクション:**
    1. **どのAPIも呼び出さない。**
    2. ユーザーの現在の発言 (`<prompt>`) を「統一会話履歴」に追加する。
    3. CLIには何も表示せず（または改行のみ行い）、次の入力待ちに戻る。
    - *このメモは、次に `@gemini` 等が呼ばれた際に文脈として渡される。*

## 3. 必須要件

- **言語:** Python 3.x
- **ライブラリ:**
    - `openai` (ChatGPT用)
    - `google-generativeai` (Gemini用)
- **設定:** APIキー (OpenAI, Google AI) は環境変数または設定ファイル (`.env`など) から読み込む。

## 4. 処理フロー（擬似コード）

```python
history = [] # 統一会話履歴

while True:
    prompt = input("> ")

    if prompt == "exit" or prompt == "quit":
        break

    # 履歴にユーザー発言を追加
    history.append({"role": "user", "content": prompt})

    if prompt.startswith("@gemini"):
        # @gemini の処理
        response_g = call_gemini_api(history)
        print(f"[Gemini]: {response_g}")
        history.append({"role": "gemini", "content": response_g})

    elif prompt.startswith("@chatgpt"):
        # @chatgpt の処理
        response_c = call_chatgpt_api(history)
        print(f"[ChatGPT]: {response_c}")
        history.append({"role": "chatgpt", "content": response_c})

    elif prompt.startswith("@all"):
        # @all の処理 (並列実行が望ましいが、直列でも可)
        response_g = call_gemini_api(history)
        response_c = call_chatgpt_api(history)
        
        print(f"[Gemini]: {response_g}")
        history.append({"role": "gemini", "content": response_g})
        
        print(f"[ChatGPT]: {response_c}")
        history.append({"role": "chatgpt", "content": response_c})

    else:
        # メンションなし (思考メモ)
        # APIは呼ばず、履歴追加のみ（既に行われている）
        pass
