"""
Data structures shared across the agent pipeline, plus the configurable
limits that prevent runaway loops (spec section 8: "Do not create an
infinite autonomous loop").
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import settings


class TaskIntent(str, Enum):
    CALCULATION = "CALCULATION"
    CSV_OR_DATA_ANALYSIS = "CSV_OR_DATA_ANALYSIS"
    FILE_READ = "FILE_READ"
    KNOWLEDGE_QUERY = "KNOWLEDGE_QUERY"
    GENERAL = "GENERAL"


class FailureCategory(str, Enum):
    """Deterministic classification of why a task failed - stored on the
    Experience so future similar failures can be recognized, and shown in
    the UI instead of a raw exception string."""

    TIMEOUT = "TIMEOUT"
    TOOL_ERROR = "TOOL_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    FILE_ERROR = "FILE_ERROR"
    PYTHON_EXECUTION_ERROR = "PYTHON_EXECUTION_ERROR"
    KNOWLEDGE_RETRIEVAL_ERROR = "KNOWLEDGE_RETRIEVAL_ERROR"
    LLM_ERROR = "LLM_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class AgentLimits:
    max_iterations: int = field(default_factory=lambda: settings.max_iterations)
    max_tool_calls: int = field(default_factory=lambda: settings.max_tool_calls)
    max_retries: int = field(default_factory=lambda: settings.max_retries)
    tool_timeout_seconds: int = field(default_factory=lambda: settings.tool_timeout_seconds)


@dataclass
class ToolCallRecord:
    tool_name: str
    input: dict
    success: bool
    output: object = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class PlanStep:
    tool_name: str
    tool_input: dict
    reason: str


@dataclass
class Plan:
    intent: TaskIntent
    steps: list[PlanStep]
    reasoning: str


@dataclass
class EvaluationResult:
    status: str  # "SUCCESS" | "PARTIAL_SUCCESS" | "FAILURE"
    score: float
    confidence: float
    reason: str
    lesson: str


@dataclass
class TraceStep:
    """One recorded step in the agent's execution trace (spec section 2).

    Never put secrets/API keys in `metadata` - the trace is meant to be
    exposed via GET /api/tasks/{id}/trace.
    """

    name: str
    status: str  # "RUNNING" | "DONE" | "FAILED"
    start_time: float
    end_time: float | None = None
    duration_seconds: float | None = None
    metadata: dict = field(default_factory=dict)
    error: str | None = None


class TraceRecorder:
    """Small helper the agent uses to time and record each pipeline step."""

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []
        self._open: TraceStep | None = None

    def start(self, name: str, **metadata) -> None:
        self._open = TraceStep(name=name, status="RUNNING", start_time=time.monotonic(), metadata=metadata)
        self.steps.append(self._open)

    def finish(self, *, error: str | None = None, **metadata) -> None:
        if self._open is None:
            return
        self._open.end_time = time.monotonic()
        self._open.duration_seconds = round(self._open.end_time - self._open.start_time, 4)
        self._open.status = "FAILED" if error else "DONE"
        self._open.error = error
        self._open.metadata.update(metadata)
        self._open = None

    def as_dicts(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "status": s.status,
                "duration_seconds": s.duration_seconds,
                "metadata": s.metadata,
                "error": s.error,
            }
            for s in self.steps
        ]


# -- deterministic failure categorization ------------------------------------

_CATEGORY_PATTERNS: list[tuple[FailureCategory, re.Pattern]] = [
    (FailureCategory.TIMEOUT, re.compile(r"timed?\s*out", re.IGNORECASE)),
    (FailureCategory.FILE_ERROR, re.compile(r"no such file or directory|file not found|escapes the uploads|unsupported file type", re.IGNORECASE)),
    (FailureCategory.PYTHON_EXECUTION_ERROR, re.compile(r"sandbox policy|traceback|nameerror|typeerror|valueerror", re.IGNORECASE)),
    (FailureCategory.INVALID_INPUT, re.compile(r"no expression provided|no filename provided|no query provided|no code provided", re.IGNORECASE)),
    (FailureCategory.KNOWLEDGE_RETRIEVAL_ERROR, re.compile(r"knowledge base", re.IGNORECASE)),
    (FailureCategory.LLM_ERROR, re.compile(r"api call failed|api_key is not set", re.IGNORECASE)),
]

_TOOL_TO_DEFAULT_CATEGORY = {
    "python_executor": FailureCategory.PYTHON_EXECUTION_ERROR,
    "file_inspector": FailureCategory.FILE_ERROR,
    "knowledge_search": FailureCategory.KNOWLEDGE_RETRIEVAL_ERROR,
    "calculator": FailureCategory.INVALID_INPUT,
}


def categorize_failure(tool_calls: list[ToolCallRecord]) -> FailureCategory:
    """Best-effort, deterministic classification from the error text and
    which tool produced it. Regex-based, not ML-based - explainable, and
    matches the same failed-call text every time."""
    failed = [c for c in tool_calls if not c.success]
    if not failed:
        return FailureCategory.UNKNOWN

    combined_error = " | ".join(c.error or "" for c in failed)
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(combined_error):
            return category

    # Fall back to a per-tool default if no message pattern matched.
    for call in failed:
        default = _TOOL_TO_DEFAULT_CATEGORY.get(call.tool_name)
        if default:
            return default
    return FailureCategory.TOOL_ERROR


# -- bounded, explainable confidence estimate --------------------------------


def compute_agent_confidence(
    *,
    evaluation: EvaluationResult,
    tool_calls: list[ToolCallRecord],
    retries: int,
    top_memory_similarity: float | None,
) -> float:
    """A bounded 0.0-1.0 confidence estimate built from concrete signals,
    not just whatever number the LLM evaluator happened to output (spec
    section 8: "Confidence should NOT simply be whatever number the LLM
    generates"). Labeled in the UI as an *estimate*, not a certainty.
    """
    score = 0.0

    if evaluation.status == "SUCCESS":
        score += 0.45
    elif evaluation.status == "PARTIAL_SUCCESS":
        score += 0.2

    if tool_calls:
        success_ratio = sum(1 for c in tool_calls if c.success) / len(tool_calls)
        score += 0.25 * success_ratio
    else:
        score += 0.1  # answered from reasoning only - inherently less certain

    score -= min(retries, 3) * 0.08

    if top_memory_similarity is not None:
        score += 0.15 * min(top_memory_similarity, 1.0)

    score += 0.15 * min(evaluation.confidence, 1.0) * 0.5  # evaluator's own read, damped

    return round(max(0.0, min(1.0, score)), 3)


@dataclass
class AgentRunResult:
    task_id: str
    final_answer: str
    plan: Plan
    tool_calls: list[ToolCallRecord]
    evaluation: EvaluationResult
    retries: int
    retrieved_experiences: list[str]  # human-readable summaries, for transparency in the UI
    activity_log: list[str]
    experience_id: str | None = None
    trace: list[dict] = field(default_factory=list)
    agent_confidence: float = 0.0
    failure_category: str | None = None