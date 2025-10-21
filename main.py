import os
import google.generativeai as genai
from dotenv import load_dotenv
import openai

load_dotenv() # Load environment variables from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
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

def call_gemini_api(history):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        gemini_history = format_history_for_gemini(history)
        response_stream = model.generate_content(gemini_history, stream=True)
        full_response = []
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end='', flush=True)
                full_response.append(chunk.text)
        print() # Newline after streaming
        return "".join(full_response)
    except genai.types.BlockedPromptException as e:
        return f"Gemini API Error: Prompt was blocked due to safety concerns. Details: {e}"
    except Exception as e:
        return f"Gemini API Error: An unexpected error occurred: {e}"

def call_chatgpt_api(history):
    try:
        if not OPENAI_API_KEY:
            return "ChatGPT API Error: OPENAI_API_KEYが設定されていません。"
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        chatgpt_history = format_history_for_chatgpt(history)
        response_stream = client.chat.completions.create(
            model=CHATGPT_MODEL, # Use configurable model
            messages=chatgpt_history,
            stream=True
        )
        full_response = []
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
                full_response.append(chunk.choices[0].delta.content)
        print() # Newline after streaming
        return "".join(full_response)
    except openai.APIError as e:
        return f"ChatGPT API Error: OpenAI APIからエラーが返されました: {e}"
    except openai.APITimeoutError as e:
        return f"ChatGPT API Error: リクエストがタイムアウトしました: {e}"
    except openai.APIConnectionError as e:
        return f"ChatGPT API Error: APIへの接続に失敗しました: {e}"
    except Exception as e:
        return f"ChatGPT API Error: 予期せぬエラーが発生しました: {e}"

def main():
    history = []

    while True:
        prompt = input("> ")

        if prompt.lower() in ["exit", "quit"]:
            break

        # Add user's prompt to the unified conversation history
        # The 'role' will be 'user' for all user inputs, regardless of mention.
        history.append({"role": "user", "content": prompt})
        print(f"DEBUG: User input added to history. Current length: {len(history)}")

        if prompt.startswith("@gemini"):
            print("DEBUG: Routing to Gemini API...")
            response_g = call_gemini_api(history)
            history.append({"role": "gemini", "content": response_g})
            print(f"[Gemini]: {response_g}")

        elif prompt.startswith("@chatgpt"):
            print("DEBUG: Routing to ChatGPT API...")
            response_c = call_chatgpt_api(history)
            history.append({"role": "chatgpt", "content": response_c})
            print(f"[ChatGPT]: {response_c}")

        elif prompt.startswith("@all"):
            print("DEBUG: Routing to both Gemini and ChatGPT APIs...")
            response_g = call_gemini_api(history)
            response_c = call_chatgpt_api(history)

            history.append({"role": "gemini", "content": response_g})
            history.append({"role": "chatgpt", "content": response_c})
            print(f"[Gemini]: {response_g}")
            print(f"[ChatGPT]: {response_c}")

        else:
            print("DEBUG: No mention, just adding to history (thought memo).")
            # No API call, history already updated with user prompt.


if __name__ == "__main__":
    main()
