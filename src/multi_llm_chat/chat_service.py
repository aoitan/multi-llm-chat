"""ChatService - business logic layer for chat operations

This module encapsulates core chat logic (mention parsing, LLM routing, history management)
independent of UI implementation. This allows both CLI and WebUI to share
the same business logic without duplication.
"""

import asyncio
import logging
import threading
import time

from .core import AgenticLoopResult, execute_with_tools_stream
from .history import HistoryStore
from .llm_provider import create_provider
from .mcp import get_mcp_manager

logger = logging.getLogger(__name__)


def parse_mention(message):
    """Parse mention from user message

    Args:
        message: User message string

    Returns:
        str or None: "gemini", "chatgpt", "all", or None if no mention
    """
    msg_stripped = message.strip()
    if msg_stripped.startswith("@gemini"):
        return "gemini"
    elif msg_stripped.startswith("@chatgpt"):
        return "chatgpt"
    elif msg_stripped.startswith("@all"):
        return "all"
    return None


ASSISTANT_LABELS = {
    "assistant": "**Assistant:**\n",
    "gemini": "**Gemini:**\n",
    "chatgpt": "**ChatGPT:**\n",
}

# Display name mapping for consistent label formatting
PROVIDER_DISPLAY_NAMES = {
    "gemini": "Gemini",
    "chatgpt": "ChatGPT",
}


class _AutosaveDebouncer:
    """Debounce autosave requests and persist only the latest state."""

    def __init__(self, store, user_id, get_state, min_interval_sec=2.0, clock=None, sleep=None):
        self._store = store
        self._user_id = user_id
        self._get_state = get_state
        self._min_interval_sec = float(min_interval_sec)
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._last_saved_at = None
        self._pending_task = None
        self._pending_loop = None
        self._pending_timer = None
        self._pending_scheduled = False
        self._pending_token = 0
        self._lock = threading.Lock()

    def request_save(self):
        """Request autosave with debounce."""
        with self._lock:
            last_saved_at = self._last_saved_at
            if last_saved_at is not None:
                now = self._clock()
                elapsed = now - last_saved_at
            else:
                elapsed = None

            if last_saved_at is None:
                immediate = True
                delay = None
                token = None
            elif elapsed >= self._min_interval_sec:
                immediate = True
                delay = None
                token = None
            elif self._has_pending_locked():
                return
            else:
                immediate = False
                delay = max(0.0, self._min_interval_sec - elapsed)
                self._pending_scheduled = True
                self._pending_token += 1
                token = self._pending_token

        if immediate:
            self.cancel_pending()
            self._save_now()
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._schedule_threaded_save(delay, token)
            return

        try:
            task = loop.create_task(self._delayed_save(delay, token))
        except Exception:
            with self._lock:
                self._pending_scheduled = False
            raise
        with self._lock:
            self._pending_task = task
            self._pending_loop = loop
            self._pending_scheduled = False

    def _save_now(self):
        try:
            system_prompt, turns = self._get_state()
            self._store.save_autosave(self._user_id, system_prompt, turns)
            with self._lock:
                self._last_saved_at = self._clock()
        except Exception as exc:
            logger.warning("Autosave failed for user '%s': %s", self._user_id, exc)

    async def _delayed_save(self, delay, token):
        try:
            await self._sleep(delay)
            if not self._is_token_active(token):
                return
            self._save_now()
        except Exception as exc:
            logger.warning("Autosave delayed-save failed for user '%s': %s", self._user_id, exc)
        finally:
            current = asyncio.current_task()
            with self._lock:
                if self._pending_task is current:
                    self._pending_task = None
                    self._pending_loop = None

    def _schedule_threaded_save(self, delay, token):
        timer_holder = {}

        def _run():
            timer = timer_holder["timer"]
            try:
                if not self._is_token_active(token):
                    return
                self._save_now()
            finally:
                with self._lock:
                    if self._pending_timer is timer:
                        self._pending_timer = None

        timer = threading.Timer(delay, _run)
        timer.daemon = True
        timer_holder["timer"] = timer
        with self._lock:
            self._pending_timer = timer
            self._pending_scheduled = False
        timer.start()

    def _has_pending(self):
        with self._lock:
            return self._has_pending_locked()

    def _has_pending_locked(self):
        if self._pending_scheduled:
            return True
        if self._pending_task is not None and not self._pending_task.done():
            return True
        if self._pending_timer is not None and self._pending_timer.is_alive():
            return True
        return False

    def _is_token_active(self, token):
        with self._lock:
            return token == self._pending_token

    def cancel_pending(self):
        """Cancel pending delayed save task if exists."""
        task = None
        loop = None
        timer = None
        with self._lock:
            if self._pending_task is not None and not self._pending_task.done():
                task = self._pending_task
                loop = self._pending_loop
            self._pending_task = None
            self._pending_loop = None

            if self._pending_timer is not None and self._pending_timer.is_alive():
                timer = self._pending_timer
            self._pending_timer = None
            self._pending_scheduled = False
            self._pending_token += 1

        if task is not None:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None

            if loop is not None and loop is current_loop:
                task.cancel()
            elif loop is not None and loop.is_running():
                loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()

        if timer is not None:
            timer.cancel()

    def flush_now(self):
        """Force autosave now and clear pending debounce work."""
        self.cancel_pending()
        self._save_now()


class ChatService:
    """Business logic layer for chat operations

    Encapsulates core chat logic (mention parsing, LLM routing, history management)
    independent of UI implementation. This allows both CLI and WebUI to share
    the same business logic without duplication.

    Attributes:
        display_history: UI-friendly history format [[user_msg, assistant_msg], ...]
        logic_history: API-friendly history format [{"role": "user", "content": "..."}]
        system_prompt: System prompt text for LLM context
        gemini_provider: Gemini LLM provider instance
        chatgpt_provider: ChatGPT LLM provider instance
        mcp_client: Optional MCP client for tool execution
    """

    def __init__(
        self,
        display_history=None,
        logic_history=None,
        system_prompt="",
        gemini_provider=None,
        chatgpt_provider=None,
        mcp_client=None,
        autosave_store=None,
        autosave_user_id=None,
        autosave_min_interval_sec=2.0,
        autosave_clock=None,
        autosave_sleep=None,
    ):
        """Initialize ChatService with optional existing state and providers

        Args:
            display_history: Optional existing display history
            logic_history: Optional existing logic history
            system_prompt: Optional system prompt (default: "")
            gemini_provider: Optional Gemini provider instance (lazy-created if None)
            chatgpt_provider: Optional ChatGPT provider instance (lazy-created if None)
            mcp_client: Optional MCP client for tool execution.
                       If None, will use global MCPServerManager from get_mcp_manager()
        """
        self.display_history = display_history if display_history is not None else []
        self.logic_history = logic_history if logic_history is not None else []
        self.system_prompt = system_prompt
        self._autosave_store = autosave_store
        self._autosave_user_id = None
        self._autosave_min_interval_sec = autosave_min_interval_sec
        self._autosave_clock = autosave_clock
        self._autosave_sleep = autosave_sleep
        self._autosave_debouncer = None

        # Store injected providers or None for lazy initialization
        self._gemini_provider = gemini_provider
        self._chatgpt_provider = chatgpt_provider

        # Use provided mcp_client or fall back to global MCPServerManager
        if mcp_client is not None:
            self.mcp_client = mcp_client
        else:
            # Try to get global MCPServerManager
            self.mcp_client = get_mcp_manager()

        if autosave_user_id:
            self.configure_autosave(
                autosave_user_id,
                store=autosave_store,
                min_interval_sec=autosave_min_interval_sec,
            )

    @property
    def gemini_provider(self):
        """Lazy-initialized Gemini provider"""
        if self._gemini_provider is None:
            self._gemini_provider = create_provider("gemini")
        return self._gemini_provider

    @property
    def chatgpt_provider(self):
        """Lazy-initialized ChatGPT provider"""
        if self._chatgpt_provider is None:
            self._chatgpt_provider = create_provider("chatgpt")
        return self._chatgpt_provider

    def _handle_api_error(self, error, provider_name):
        """Handle API errors in a consistent way

        Args:
            error: Exception that occurred
            provider_name: Name of the provider ("gemini" or "chatgpt")
        """
        provider_title = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.capitalize())
        label = ASSISTANT_LABELS.get(provider_name, f"**{provider_title}:**\n")

        if isinstance(error, ValueError):
            # API key errors
            error_msg = f"[System: エラー - {str(error)}]"
        else:
            # Other API errors (network, blocked prompts, etc.)
            error_msg = f"[System: {provider_title} APIエラー - {str(error)}]"

        # Ensure display_history ends with an assistant entry before writing error
        if not self.display_history or self.display_history[-1].get("role") != "assistant":
            self.display_history.append({"role": "assistant", "content": ""})
        current_content = self.display_history[-1]["content"]
        # Preserve partial response if streaming had already started (content beyond the label)
        if current_content in (label, ""):
            self.display_history[-1]["content"] = f"{label}{error_msg}"
        else:
            self.display_history[-1]["content"] = current_content + f"\n\n{error_msg}"
        self.logic_history.append(
            {
                "role": provider_name,
                "content": [{"type": "text", "content": error_msg}],
            }
        )

    def configure_autosave(self, user_id, store=None, min_interval_sec=None):
        """Configure autosave behavior for this service instance."""
        if not user_id:
            if self._autosave_debouncer is not None:
                self._autosave_debouncer.cancel_pending()
            self._autosave_user_id = None
            self._autosave_debouncer = None
            return

        resolved_store = store if store is not None else self._autosave_store
        if resolved_store is None:
            resolved_store = HistoryStore()

        resolved_interval = (
            self._autosave_min_interval_sec
            if min_interval_sec is None
            else min_interval_sec
        )

        if (
            self._autosave_debouncer is not None
            and self._autosave_user_id == user_id
            and resolved_store is self._autosave_store
            and resolved_interval == self._autosave_min_interval_sec
        ):
            return

        if self._autosave_debouncer is not None:
            self._autosave_debouncer.cancel_pending()

        self._autosave_store = resolved_store
        self._autosave_user_id = user_id
        self._autosave_min_interval_sec = resolved_interval
        self._autosave_debouncer = _AutosaveDebouncer(
            store=self._autosave_store,
            user_id=user_id,
            get_state=lambda: (self.system_prompt, self.logic_history),
            min_interval_sec=resolved_interval,
            clock=self._autosave_clock,
            sleep=self._autosave_sleep,
        )

    def request_autosave(self):
        """Request autosave if configured."""
        if self._autosave_debouncer is not None:
            self._autosave_debouncer.request_save()

    def flush_autosave(self):
        """Flush pending autosave immediately if configured."""
        if self._autosave_debouncer is not None:
            self._autosave_debouncer.flush_now()

    async def process_message(self, user_message, tools=None):
        """Process user message and generate LLM responses

        This is an async generator function that yields intermediate states
        for streaming UI updates.

        Args:
            user_message: User's input message
            tools: Optional list of tools for the LLM

        tuple:
            tuple: (display_history, logic_history, chunk) after each update
        """
        mention = parse_mention(user_message)

        # Add user message to histories (structured format)
        user_entry = {"role": "user", "content": [{"type": "text", "content": user_message}]}
        self.logic_history.append(user_entry)
        self.display_history.append({"role": "user", "content": user_message})
        self.request_autosave()
        yield self.display_history, self.logic_history, {"type": "text", "content": ""}

        # If no mention, treat as memo (no LLM call)
        if mention is None:
            return

        # For @all, create lightweight snapshot; execute_with_tools_stream will deep-copy internally
        history_at_start = list(self.logic_history) if mention == "all" else None

        # Process models
        models_to_call = []
        if mention == "all":
            models_to_call = ["gemini", "chatgpt"]
        else:
            models_to_call = [mention]

        for model_name in models_to_call:
            provider = self.gemini_provider if model_name == "gemini" else self.chatgpt_provider
            label = ASSISTANT_LABELS[model_name]

            # Add assistant entry for this model's response
            self.display_history.append({"role": "assistant", "content": label})

            yield self.display_history, self.logic_history, {"type": "text", "content": ""}

            # Prepare input history for this model
            # execute_with_tools_stream will deep-copy it internally
            if mention == "all":
                input_history = history_at_start
            else:
                input_history = self.logic_history

            any_yielded = False
            result = None

            try:
                # Execute with tools and stream chunks in real-time
                async for item in execute_with_tools_stream(
                    provider,
                    input_history,
                    self.system_prompt,
                    mcp_client=self.mcp_client,
                    tools=tools,
                ):
                    # Check if this is the final result
                    if isinstance(item, AgenticLoopResult):
                        result = item
                        continue

                    # This is a streaming chunk
                    chunk = item
                    any_yielded = True
                    chunk_type = chunk.get("type")
                    content = chunk.get("content", "")

                    if chunk_type == "text":
                        if content:
                            self.display_history[-1]["content"] += content

                    yield self.display_history, self.logic_history, chunk

                # Update history with delta
                if result is None:
                    logger.warning(
                        "execute_with_tools_stream did not yield AgenticLoopResult; "
                        "history not updated"
                    )
                elif mention == "all":
                    # For @all, extend logic_history with new entries
                    self.logic_history.extend(result.history_delta)
                else:
                    # For specific mention, input_history is self.logic_history
                    # execute_with_tools_stream no longer mutates history,
                    # so we extend it explicitly
                    input_history.extend(result.history_delta)

                if not any_yielded:
                    error_message = (
                        f"[System: {model_name.capitalize()}からの応答がありませんでした]"
                    )
                    self.display_history[-1]["content"] += error_message
                    new_entry = {
                        "role": model_name,
                        "content": [{"type": "text", "content": error_message}],
                    }
                    self.logic_history.append(new_entry)

            except (ValueError, Exception) as e:
                self._handle_api_error(e, model_name)
                yield self.display_history, self.logic_history, {"type": "error", "content": str(e)}

            self.request_autosave()
            yield self.display_history, self.logic_history, {"type": "text", "content": ""}

    def set_system_prompt(self, prompt):
        """Update system prompt

        Args:
            prompt: New system prompt text
        """
        self.system_prompt = prompt
        self.request_autosave()

    def append_tool_results(self, tool_results):
        """Append tool execution results to logic history.

        Args:
            tool_results: List of tool result dicts with name/content and optional tool_call_id.
        """
        if not tool_results:
            return

        content_parts = []
        for result in tool_results:
            if not isinstance(result, dict):
                logger.warning(
                    "Invalid tool result type: %s (expected dict), skipping",
                    type(result).__name__,
                )
                continue
            if result.get("type") == "tool_result":
                content_parts.append(result)
                continue
            content_parts.append(
                {
                    "type": "tool_result",
                    "name": result.get("name"),
                    "content": result.get("content", ""),
                    "tool_call_id": result.get("tool_call_id"),
                }
            )

        if content_parts:
            self.logic_history.append(
                {
                    "role": "tool",
                    "content": content_parts,
                }
            )
