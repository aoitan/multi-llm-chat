from typing import List, Dict, Any, Tuple, Union, Optional

LLM_ROLES = {"gemini", "chatgpt"}


def get_provider_name_from_model(model_name: str) -> str:
    """Get provider name from model name

    Args:
        model_name: Model identifier

    Returns:
        str: Provider name ("gemini" or "chatgpt")
    """
    model_lower = model_name.lower()
    if "gpt" in model_lower or "chatgpt" in model_lower:
        return "chatgpt"
    return "gemini"


def prepare_request(
    history: List[Dict[str, Any]], system_prompt: str, model_name: str
) -> Union[List[Dict[str, Any]], Tuple[Optional[str], List[Dict[str, Any]]]]:
    """Prepare API request with system prompt and history"""
    if "gemini" in model_name.lower():
        # For Gemini, return tuple (system_prompt, history)
        # Only include system_prompt if it's not empty or whitespace-only
        if system_prompt and system_prompt.strip():
            return (system_prompt, history)
        else:
            return (None, history)
    else:
        # For OpenAI-compatible models, add system message to history
        # Only add system message if prompt is not empty
        if system_prompt and system_prompt.strip():
            return [{"role": "system", "content": system_prompt}] + history
        else:
            return history
