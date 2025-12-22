import asyncio
import time
from typing import Dict, List, Optional


class PersonaContext:
    """Per-persona conversation context with per-room history and memory.

    - Keeps isolated history per (persona, room)
    - Enforces a bounded history size (max_turns)
    - Provides a simple memory dict per room for lightweight state
    """

    def __init__(self, persona_slug: str, max_turns: int = 40):
        self.persona_slug = persona_slug
        self._history_by_room: Dict[str, List[Dict[str, str]]] = {}
        self._memory_by_room: Dict[str, Dict[str, object]] = {}
        self._lock = asyncio.Lock()
        self._max_turns = max_turns

    def get_room_history(self, room_id: str) -> List[Dict[str, str]]:
        """Return the history list for a given room (create if missing)."""
        return self._history_by_room.setdefault(room_id, [])

    def get_room_memory(self, room_id: str) -> Dict[str, object]:
        """Return the memory dict for a given room (create if missing)."""
        return self._memory_by_room.setdefault(room_id, {})

    async def add_turn(self, room_id: str, role: str, text: str) -> None:
        """Append a turn to room history and enforce max size.

        Role is expected to be 'user' or 'assistant'.
        """
        async with self._lock:
            history = self._history_by_room.setdefault(room_id, [])
            history.append({
                "role": role,
                "content": text,
                "timestamp": time.time(),
            })
            # Enforce bounded size: keep last N turns
            if len(history) > self._max_turns:
                # Trim from the front
                excess = len(history) - self._max_turns
                del history[:excess]

    async def clear_room(self, room_id: str) -> None:
        """Clear history and memory for a room (optional utility)."""
        async with self._lock:
            self._history_by_room.pop(room_id, None)
            self._memory_by_room.pop(room_id, None)


class PersonaContextManager:
    """Async-safe manager returning PersonaContext instances per persona slug."""

    def __init__(self):
        self._contexts: Dict[str, PersonaContext] = {}
        self._lock = asyncio.Lock()

    async def get_context(self, persona_slug: str) -> PersonaContext:
        async with self._lock:
            ctx = self._contexts.get(persona_slug)
            if ctx is None:
                ctx = PersonaContext(persona_slug)
                self._contexts[persona_slug] = ctx
            return ctx


# Singleton accessor used by AIOrchestrator
_manager: Optional[PersonaContextManager] = None


async def get_persona_context_manager() -> PersonaContextManager:
    global _manager
    if _manager is None:
        _manager = PersonaContextManager()
    return _manager
