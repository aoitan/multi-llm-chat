import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import core
from .history_utils import content_to_text, normalize_history_turns

# Schema version constants for history storage
SCHEMA_VERSION_LEGACY = 1  # String-based content (deprecated in v2.0.0)
SCHEMA_VERSION_STRUCTURED = 2  # List[Dict] content (current)


def _default_base_dir() -> Path:
    """Resolve the base directory for history storage."""
    env_dir = os.getenv("CHAT_HISTORY_DIR")
    if env_dir:
        return Path(env_dir)

    xdg_data = os.getenv("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "multi_llm_chat" / "chat_histories"

    # User home fallback
    return Path.home() / ".multi_llm_chat" / "chat_histories"


def sanitize_name(name: str) -> str:
    """Sanitize user-provided identifiers for filesystem safety."""
    raw = (name or "").strip()
    if raw in {"", ".", ".."} or os.path.sep in raw or (os.path.altsep and os.path.altsep in raw):
        raise ValueError("Invalid name")

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if sanitized in {"", ".", ".."}:
        raise ValueError("Invalid name")
    return sanitized


def reset_history():
    """Return an empty conversation history."""
    return []


def get_llm_response(history, index):
    """指定インデックスのLLM応答本文を取得する（最新が0）。

    Args:
        history: 会話履歴のリスト。各エントリは role と content を持つ辞書。
        index: 取得する応答のインデックス（0が最新）。

    Returns:
        str: 指定されたインデックスのLLM応答本文。

    Raises:
        IndexError: インデックスが負、または該当するLLM応答が存在しない場合。
    """
    if index < 0:
        raise IndexError("index must be non-negative")

    responses = [
        content_to_text(entry.get("content", ""), include_tool_data=False)
        for entry in reversed(history or [])
        if entry.get("role") in core.LLM_ROLES
    ]

    try:
        return responses[index]
    except IndexError as exc:
        raise IndexError("LLM response not found for the given index") from exc


class HistoryStore:
    """Filesystem-backed store for chat histories."""

    def __init__(self, base_dir: Optional[Path] = None):
        resolved_base = base_dir if base_dir is not None else _default_base_dir()
        self.base_dir = Path(resolved_base)

    def _user_dir(self, user_id: str) -> Path:
        return self.base_dir / sanitize_name(user_id)

    def _history_path(self, user_id: str, display_name: str) -> Path:
        return self._user_dir(user_id) / f"{sanitize_name(display_name)}.json"

    def history_exists(self, user_id: str, display_name: str) -> bool:
        """Check if a history file already exists."""
        return self._history_path(user_id, display_name).exists()

    def list_histories(self, user_id: str) -> List[str]:
        """Return sorted display names for a user."""
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return []

        display_names: List[str] = []
        for path in sorted(user_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            name = data.get("display_name")
            if name:
                display_names.append(name)

        return sorted(display_names)

    def save_history(self, user_id: str, display_name: str, system_prompt: str, turns):
        """Persist a conversation to disk with normalization."""
        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        path = self._history_path(user_id, display_name)
        overwritten = path.exists()

        now = datetime.now(timezone.utc).isoformat()
        created_at = now
        if overwritten:
            try:
                existing = json.loads(path.read_text())
                created_at = existing.get("metadata", {}).get("created_at", now)
            except (json.JSONDecodeError, OSError):
                created_at = now

        # Normalize turns to structured format before saving
        if turns:
            turns = normalize_history_turns(turns)

        payload = {
            "display_name": display_name,
            "system_prompt": system_prompt,
            "turns": turns,
            "metadata": {
                "schema_version": SCHEMA_VERSION_STRUCTURED,
                "created_at": created_at,
                "updated_at": now,
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return {"path": path, "overwritten": overwritten}

    def load_history(self, user_id: str, display_name: str):
        """Load a conversation by display name."""
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            raise FileNotFoundError("User directory not found")

        target_sanitized = sanitize_name(display_name)
        for path in user_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            if data.get("display_name") == display_name or path.stem == target_sanitized:
                metadata = data.get("metadata", {})
                metadata.setdefault("schema_version", 1)
                if "created_at" not in metadata:
                    metadata["created_at"] = datetime.now(timezone.utc).isoformat()
                if "updated_at" not in metadata:
                    metadata["updated_at"] = metadata["created_at"]
                data["metadata"] = metadata
                data.setdefault("system_prompt", "")
                data["turns"] = normalize_history_turns(data.get("turns", []))
                return data

        raise FileNotFoundError("History not found")
