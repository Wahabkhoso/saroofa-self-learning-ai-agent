"""
Tool system.

Every tool implements this same small interface, so the agent's tool
selection / execution code never needs to know about a specific tool.
Adding a new tool = write a class, add one line to the registry.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None
    duration_seconds: float = 0.0


class Tool(ABC):
    name: str
    description: str
    # Metadata for the Tool Registry (GET /api/tools) - every concrete
    # tool should set these; risk_level is a plain, explainable label,
    # not a computed score.
    risk_level: str = "LOW"  # "LOW" | "MEDIUM" | "HIGH"
    timeout_seconds: int | None = None
    enabled: bool = True
    input_schema: dict = {}
    output_schema: dict = {}

    @abstractmethod
    def _run(self, **kwargs) -> ToolResult:
        ...

    def run(self, **kwargs) -> ToolResult:
        start = time.monotonic()
        try:
            result = self._run(**kwargs)
        except Exception as exc:  # noqa: BLE001 - tools must never raise past this point
            result = ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
        result.duration_seconds = round(time.monotonic() - start, 4)
        return result

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list_metadata(self) -> list[dict]:
        return [t.metadata() for t in self._tools.values()]

    def describe_all(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())