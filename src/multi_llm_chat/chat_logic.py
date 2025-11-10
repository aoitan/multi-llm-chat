import os
import google.generativeai as genai
from dotenv import load_dotenv
import openai

load_dotenv() # Load environment variables from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo")

def _configure_gemini():
    """Configure the Gemini SDK if an API key is available."""
    if not GOOGLE_API_KEY:
        return False
    genai.configure(api_key=GOOGLE_API_KEY)
    return True

def format_history_for_gemini(history):
    # Gemini API expects a list of dicts with 'role' and 'parts'
    # 'user' role for user input, 'model' for AI responses
    gemini_history = []
    for entry in history:
        role = "user" if entry["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [entry["content"]]})
    return gemini_history

def format_history_for_chatgpt(history):
    # ChatGPT API expects a list of dicts with 'role' and 'content'
    # 'user' for user, 'assistant' for AI
    chatgpt_history = []
    for entry in history:
        role = "user" if entry["role"] == "user" else "assistant"
        chatgpt_history.append({"role": role, "content": entry["content"]})
    return chatgpt_history

def list_gemini_models():
    if not _configure_gemini():
        print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
        return

    print("利用可能なGeminiモデル:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  - {m.name}")

def call_gemini_api(history):
    if not _configure_gemini():
        yield "Gemini API Error: GOOGLE_API_KEY not found in environment variables or .env file."
        return

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        gemini_history = format_history_for_gemini(history)
        # print(f"DEBUG: Gemini API request history: {gemini_history}", flush=True)
        response_stream = model.generate_content(gemini_history, stream=True)
        for chunk in response_stream:
            yield chunk
    except genai.types.BlockedPromptException as e:
        yield f"Gemini API Error: Prompt was blocked due to safety concerns. Details: {e}"
    except Exception as e:
        print(f"Gemini API Error: An unexpected error occurred: {e}")
        list_gemini_models()
        yield f"Gemini API Error: An unexpected error occurred: {e}"

def call_chatgpt_api(history):
    try:
        if not OPENAI_API_KEY:
            yield "ChatGPT API Error: OPENAI_API_KEYが設定されていません。"
            return
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        chatgpt_history = format_history_for_chatgpt(history)
        # print(f"DEBUG: ChatGPT API request history: {chatgpt_history}", flush=True)
        response_stream = client.chat.completions.create(
            model=CHATGPT_MODEL,
            messages=chatgpt_history,
            stream=True
        )
        for chunk in response_stream:
            yield chunk
    except openai.APIError as e:
        yield f"ChatGPT API Error: OpenAI APIからエラーが返されました: {e}"
    except openai.APITimeoutError as e:
        yield f"ChatGPT API Error: リクエストがタイムアウトしました: {e}"
    except openai.APIConnectionError as e:
        yield f"ChatGPT API Error: APIへの接続に失敗しました: {e}"
    except Exception as e:
        yield f"ChatGPT API Error: 予期せぬエラーが発生しました: {e}"

def _process_response_stream(stream, model_name):
    """Helper function to process and print response streams from LLMs."""
    full_response = ""
    print(f"[{model_name.capitalize()}]: ", end='', flush=True)
    
    for chunk in stream:
        text = ""
        try:
            if model_name == "gemini":
                text = chunk.text
            elif model_name == "chatgpt":
                text = chunk.choices[0].delta.content
        except (AttributeError, IndexError, TypeError, ValueError):
            if isinstance(chunk, str):
                text = chunk # Handle yielded error strings
        
        if text:
            print(text, end='', flush=True)
            full_response += text
            
    print() # Newline after the full response
    if not full_response.strip():
        print(f"[System: {model_name.capitalize()}からの応答がありませんでした。プロンプトがブロックされた可能性があります。]", flush=True)
        
    return full_response

def main():
    history = []

    while True:
        prompt = input("> ").strip()

        if prompt.lower() in ["exit", "quit"]:
            break

        history.append({"role": "user", "content": prompt})

        if prompt.startswith("@gemini"):
            gemini_stream = call_gemini_api(history)
            response_g = _process_response_stream(gemini_stream, "gemini")
            history.append({"role": "gemini", "content": response_g})

        elif prompt.startswith("@chatgpt"):
            chatgpt_stream = call_chatgpt_api(history)
            response_c = _process_response_stream(chatgpt_stream, "chatgpt")
            history.append({"role": "chatgpt", "content": response_c})

        elif prompt.startswith("@all"):
            # Call both APIs in sequence for now, can be parallelized later
            gemini_stream = call_gemini_api(history)
            response_g = _process_response_stream(gemini_stream, "gemini")
            history.append({"role": "gemini", "content": response_g})

            chatgpt_stream = call_chatgpt_api(history)
            response_c = _process_response_stream(chatgpt_stream, "chatgpt")
            history.append({"role": "chatgpt", "content": response_c})

        else:
            # Thinking memo
            pass
    return history
