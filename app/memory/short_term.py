"""
Short-term memory: what happened in *this* session only.

This is deliberately plain Python data structures held in the process -
it is not meant to survive a restart. Long-term, persistent memory is
handled separately in `long_term.py` / `vector_memory.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ShortTermTurn:
    role: str  # "user" | "agent"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ShortTermMemory:
    """Holds the state of a single in-progress or recently completed task."""

    session_id: str
    turns: list[ShortTermTurn] = field(default_factory=list)
    current_task: str | None = None
    current_plan: str | None = None
    tool_results: list[dict] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.turns.append(ShortTermTurn(role="user", content=content))
        self.current_task = content

    def add_agent_message(self, content: str) -> None:
        self.turns.append(ShortTermTurn(role="agent", content=content))

    def set_plan(self, plan: str) -> None:
        self.current_plan = plan

    def record_tool_result(self, tool_name: str, result: dict) -> None:
        self.tool_results.append({"tool": tool_name, "result": result})

    def reset_task_state(self) -> None:
        self.current_plan = None
        self.tool_results = []

    def recent_history(self, n: int = 6) -> list[ShortTermTurn]:
        return self.turns[-n:]


class SessionMemoryStore:
    """Keeps one ShortTermMemory per session_id, in process memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, ShortTermMemory] = {}

    def get_or_create(self, session_id: str) -> ShortTermMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemory(session_id=session_id)
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Process-wide singleton (fine for a single-instance FastAPI app / demo).
session_memory_store = SessionMemoryStore()
