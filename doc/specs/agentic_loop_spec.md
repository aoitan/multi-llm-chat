# Agentic Loop (ReAct) 仕様

## 1. 概要

Agentic Loopは、LLMが自律的にツールを使用し、その結果に基づいて最終的な回答を生成するための実行ループです。
いわゆる **ReAct (Reasoning + Acting)** パターンを実装しており、LLMの思考（Reasoning）、行動（Acting/Tool Use）、観測（Observation/Tool Result）のサイクルを回します。

実装は `src/multi_llm_chat/core_modules/agentic_loop.py` にあります。

## 2. 実行フロー

エージェントループは以下のステップを繰り返します。

1.  **LLM呼び出し**: 現在の会話履歴とツール定義を渡してLLMを呼び出します（ストリーミング対応）。
2.  **応答解析**:
    *   **テキストのみ**: ユーザーへの最終回答とみなし、ループを終了します。
    *   **ツール呼び出し (Tool Call)**: ツール名と引数が返された場合、ステップ3へ進みます。
3.  **ツール実行**:
    *   `MCPClient` を通じて外部ツールを実行します。
    *   実行結果（成功結果またはエラーメッセージ）を取得します。
4.  **履歴更新**:
    *   LLMの「ツール呼び出し要求」を履歴に追加します。
    *   ツールの「実行結果」を履歴に追加します（Role: `tool`）。
5.  **ループ継続**: 更新された履歴を持ってステップ1に戻ります。

```mermaid
graph TD
    Start[開始] --> CallLLM[LLM呼び出し]
    CallLLM --> Check{応答タイプ?}
    
    Check -- テキストのみ --> Finish[終了 (Final Answer)]
    Check -- ツール呼び出し --> ExecTool[MCPツール実行]
    
    ExecTool --> UpdateHist[履歴更新<br/>(Tool Call + Tool Result)]
    UpdateHist --> CheckMax{最大反復回数<br/>or<br/>タイムアウト?}
    
    CheckMax -- No --> CallLLM
    CheckMax -- Yes --> Error[終了 (Timeout/Limit)]
```

## 3. 終了条件

無限ループや過剰なリソース消費を防ぐため、以下の条件でループは強制終了します。

1.  **最大反復回数 (`max_iterations`) 到達**:
    *   デフォルト: 10回
    *   これを超えるとループを抜け、その時点までの結果を返します。
2.  **タイムアウト (`timeout`)**:
    *   デフォルト: 120秒
    *   実行開始からの経過時間が閾値を超えると `TimeoutError` を送出します。
3.  **エラー発生**:
    *   回復不可能なエラーが発生した場合。

## 4. API仕様

推奨されるメインAPIは `execute_with_tools_stream` です。

### `execute_with_tools_stream`

非同期ジェネレータとして実装されており、リアルタイムでチャンクを返却します。

```python
async def execute_with_tools_stream(
    provider: BaseProvider,
    history: List[Dict],
    mcp_client: Optional[MCPClient] = None,
    max_iterations: int = 10,
    timeout: float = 120.0,
    ...
) -> AsyncIterator[Dict | AgenticLoopResult]
```

**出力（Yield）**:
1.  **Streaming Chunks**: `{"type": "text", "content": "..."}` や `{"type": "tool_call", ...}` などの部分データ。UIのリアルタイム更新に使用。
2.  **Final Result**: 最後に `AgenticLoopResult` オブジェクトを1つだけyieldします。

### `AgenticLoopResult` (データクラス)

実行結果の不変オブジェクトです。

*   `chunks`: 生成された全チャンクのリスト
*   `history_delta`: ループ内で新たに追加された会話履歴（呼び出し元で結合するための差分）
*   `final_text`: 最終的なテキスト回答
*   `iterations_used`: 消費した反復回数
*   `timed_out`: タイムアウトしたかどうか

## 5. エラーハンドリング

*   **ツール実行エラー**:
    *   個別のツール実行が失敗した場合、エージェントループ自体は停止しません。
    *   エラーメッセージが「ツール実行結果」としてLLMにフィードバックされ、LLMはそれを踏まえてリトライや修正を試みることができます。
*   **MCPクライアント未設定**:
    *   ツール呼び出しが要求されたのに `mcp_client` が `None` の場合、エラーメッセージを履歴に追加してループを中断します。
