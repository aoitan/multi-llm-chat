import os
import google.generativeai as genai
from dotenv import load_dotenv
import openai

load_dotenv() # Load environment variables from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

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
    print("利用可能なGeminiモデル:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  - {m.name}")

def call_gemini_api(history):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        gemini_history = format_history_for_gemini(history)
        # print(f"DEBUG: Gemini API request history: {gemini_history}", flush=True)
        response_stream = model.generate_content(gemini_history, stream=True)
        # print(f"DEBUG: Gemini API response stream object: {response_stream}", flush=True)
        return response_stream
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
        # print(f"DEBUG: ChatGPT API response stream object: {response_stream}", flush=True)
        return response_stream
    except openai.APIError as e:
        yield f"ChatGPT API Error: OpenAI APIからエラーが返されました: {e}"
    except openai.APITimeoutError as e:
        yield f"ChatGPT API Error: リクエストがタイムアウトしました: {e}"
    except openai.APIConnectionError as e:
        yield f"ChatGPT API Error: APIへの接続に失敗しました: {e}"
    except Exception as e:
        yield f"ChatGPT API Error: 予期せぬエラーが発生しました: {e}"

def main():
    history = []

    while True:
        prompt = input("> ").strip()

        if prompt.lower() in ["exit", "quit"]:
            break

        history.append({"role": "user", "content": prompt})

        if prompt.startswith("#gemini"):
            print("[Gemini]: ", end='', flush=True)
            full_response = []
            gemini_response_stream = call_gemini_api(history)
            for chunk in gemini_response_stream:
                # print(f"DEBUG: Gemini chunk: {chunk}", flush=True)
                if chunk.text:
                    print(chunk.text, end='', flush=True)
                    full_response.append(chunk.text)
                elif isinstance(chunk, str): # For error messages yielded
                    print(chunk, end='', flush=True)
                    full_response.append(chunk)
            print()
            response_g = "".join(full_response)
            # print(f"DEBUG: Gemini full response collected: '{response_g}'")
            history.append({"role": "gemini", "content": response_g})

        elif prompt.startswith("#chatgpt"):
            print("[ChatGPT]: ", end='', flush=True)
            full_response = []
            chatgpt_response_stream = call_chatgpt_api(history)
            for chunk in chatgpt_response_stream:
                # print(f"DEBUG: ChatGPT chunk: {chunk}", flush=True)
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end='', flush=True)
                    full_response.append(chunk.choices[0].delta.content)
                elif isinstance(chunk, str): # For error messages yielded
                    print(chunk, end='', flush=True)
                    full_response.append(chunk)
            print()
            response_c = "".join(full_response)
            # print(f"DEBUG: ChatGPT full response collected: '{response_c}'")
            history.append({"role": "chatgpt", "content": response_c})

        elif prompt.startswith("#all"):
            # Call both APIs in sequence for now, can be parallelized later
            print("[Gemini]: ", end='', flush=True)
            full_response_g = []
            gemini_response_stream = call_gemini_api(history)
            for chunk in gemini_response_stream:
                # print(f"DEBUG: Gemini chunk (all): {chunk}", flush=True)
                if chunk.text:
                    print(chunk.text, end='', flush=True)
                    full_response_g.append(chunk.text)
                elif isinstance(chunk, str): # For error messages yielded
                    print(chunk, end='', flush=True)
                    full_response_g.append(chunk)
            print()
            response_g = "".join(full_response_g)
            # print(f"DEBUG: Gemini full response collected (all): '{response_g}'")
            history.append({"role": "gemini", "content": response_g})

            print("[ChatGPT]: ", end='', flush=True)
            full_response_c = []
            chatgpt_response_stream = call_chatgpt_api(history)
            for chunk in chatgpt_response_stream:
                # print(f"DEBUG: ChatGPT chunk (all): {chunk}", flush=True)
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end='', flush=True)
                    full_response_c.append(chunk.choices[0].delta.content)
                elif isinstance(chunk, str): # For error messages yielded
                    print(chunk, end='', flush=True)
                    full_response_c.append(chunk)
            print()
            response_c = "".join(full_response_c)
            # print(f"DEBUG: ChatGPT full response collected (all): '{response_c}'")
            history.append({"role": "chatgpt", "content": response_c})

        else:
            pass
    return history


if __name__ == "__main__":
    main()
