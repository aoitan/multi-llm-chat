import os

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

        # Placeholder for API calls and response handling (to be implemented in later stories)
        # if prompt.startswith("@gemini"):
        #     response_g = call_gemini_api(history)
        #     history.append({"role": "gemini", "content": response_g})
        #     print(f"[Gemini]: {response_g}")
        # elif prompt.startswith("@chatgpt"):
        #     response_c = call_chatgpt_api(history)
        #     history.append({"role": "chatgpt", "content": response_c})
        #     print(f"[ChatGPT]: {response_c}")
        # elif prompt.startswith("@all"):
        #     response_g = call_gemini_api(history)
        #     response_c = call_chatgpt_api(history)
        #     history.append({"role": "gemini", "content": response_g})
        #     history.append({"role": "chatgpt", "content": response_c})
        #     print(f"[Gemini]: {response_g}")
        #     print(f"[ChatGPT]: {response_c}")
        # else:
        #     # No mention, just a thought memo. History already updated.
        #     pass


if __name__ == "__main__":
    main()
