import os
import google.generativeai as genai
from dotenv import load_dotenv
# import openai # Will uncomment when actually implementing ChatGPT API

load_dotenv() # Load environment variables from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

def call_gemini_api(history):
    # Placeholder for actual Gemini API call
    print("DEBUG: Calling Gemini API with history...")
    # In a real scenario, you would format the history for Gemini API
    # and make the actual API call here.
    return "Gemini's response (placeholder)"

def call_chatgpt_api(history):
    # Placeholder for actual ChatGPT API call
    print("DEBUG: Calling ChatGPT API with history...")
    # In a real scenario, you would format the history for ChatGPT API
    # and make the actual API call here.
    return "ChatGPT's response (placeholder)"

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
            print("DEBUG: Routing to Gemini API (placeholder)")
            response_g = call_gemini_api(history)
            history.append({"role": "gemini", "content": response_g})
            print(f"[Gemini]: {response_g}")

        elif prompt.startswith("@chatgpt"):
            print("DEBUG: Routing to ChatGPT API (placeholder)")
            if not OPENAI_API_KEY:
                print("Error: OPENAI_API_KEY not found. Cannot call ChatGPT API.")
                continue
            response_c = call_chatgpt_api(history)
            history.append({"role": "chatgpt", "content": response_c})
            print(f"[ChatGPT]: {response_c}")

        elif prompt.startswith("@all"):
            print("DEBUG: Routing to both Gemini and ChatGPT APIs (placeholder)")
            response_g = call_gemini_api(history)
            
            if not OPENAI_API_KEY:
                print("Error: OPENAI_API_KEY not found. Cannot call ChatGPT API for @all.")
                response_c = "ChatGPT API call skipped due to missing API key."
            else:
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
