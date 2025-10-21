import os

def main():
    history = []

    while True:
        prompt = input("> ")

        if prompt.lower() in ["exit", "quit"]:
            break

        # For now, just add to history.
        # Actual processing based on mentions will be added in later stories.
        history.append({"role": "user", "content": prompt})
        print(f"DEBUG: Current history length: {len(history)}")
        print(f"DEBUG: Last entry: {history[-1]}")

if __name__ == "__main__":
    main()
