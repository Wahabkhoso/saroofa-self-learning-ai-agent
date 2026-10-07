"""
Task analysis and planning.

Design decision: tool *selection* is rule-based (regex/keyword
heuristics), not left to the LLM to decide freely. Why: this keeps the
agent's tool-choice behavior deterministic and testable without an API
key, and avoids a whole class of failure where the LLM "decides" to call
a tool with malformed arguments. The LLM is used downstream - in
`agent.py` - to turn tool results into a natural-language answer, and
optionally in `evaluator.py` for a qualitative correctness check. If you
want LLM-driven planning instead, this is the one file to replace; the
`Plan` / `PlanStep` contract stays the same.
"""
from __future__ import annotations

import re

from app.agent.state import Plan, PlanStep, TaskIntent
from app.core.config import settings

_CSV_FILENAME_RE = re.compile(r"(\S+\.csv)", re.IGNORECASE)
_FILENAME_RE = re.compile(r"(\S+\.(txt|md|json|csv))", re.IGNORECASE)
_AVERAGE_COLUMN_RE = re.compile(r"average\s+([a-zA-Z_ ]+)")
_FUNC_NAMES = ("sqrt", "abs", "round", "log10", "log", "sin", "cos", "tan", "floor", "ceil", "min", "max", "pi", "e")
_STRIP_PREFIXES = (
    "calculate", "compute", "what is", "what's", "how much is", "evaluate", "solve",
)


def _extract_expression(task_text: str) -> str:
    text = task_text.strip()
    lowered = text.lower()
    for prefix in _STRIP_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip(" :?")
            break

    # Keep only characters that can legally appear in a math expression, plus
    # whitelisted function names - this is what actually keeps stray English
    # words (which would otherwise look like unknown function calls) out of
    # the string handed to the calculator tool.
    func_alternation = "|".join(_FUNC_NAMES)
    token_re = re.compile(rf"({func_alternation}|[0-9.]+|[-+*/^%().])")
    tokens = token_re.findall(text.lower())
    return "".join(tokens).strip() or text

CSV_KEYWORDS = ("csv", "dataframe", "pandas", "spreadsheet")
CALC_KEYWORDS = ("calculate", "compute", "average", "sum", "total", "how much is")
KNOWLEDGE_KEYWORDS = ("document", "according to", "knowledge base", "uploaded file said")

_CALC_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CALC_KEYWORDS) + r")\b", re.IGNORECASE
)


def _has_calc_keyword(text: str) -> bool:
    """Word-boundary match, not substring - 'sum' must not match inside 'Summarize'."""
    return bool(_CALC_KEYWORD_RE.search(text))


class TaskAnalyzer:
    """Classifies free-text tasks into a small set of intents."""

    def analyze(self, task_text: str) -> TaskIntent:
        text = task_text.lower()

        if any(k in text for k in CSV_KEYWORDS) or _CSV_FILENAME_RE.search(task_text):
            return TaskIntent.CSV_OR_DATA_ANALYSIS

        if _FILENAME_RE.search(task_text) and not _has_calc_keyword(text):
            return TaskIntent.FILE_READ

        if _has_calc_keyword(text) or re.search(r"[-+*/^()]|\bsqrt\b|\bpercent\b", text):
            return TaskIntent.CALCULATION

        if any(k in text for k in KNOWLEDGE_KEYWORDS):
            return TaskIntent.KNOWLEDGE_QUERY

        return TaskIntent.GENERAL


class Planner:
    """Turns an intent + task text into a concrete, executable Plan."""

    def build_plan(self, task_text: str, intent: TaskIntent) -> Plan:
        if intent == TaskIntent.CSV_OR_DATA_ANALYSIS:
            return self._plan_csv_analysis(task_text)
        if intent == TaskIntent.CALCULATION:
            return self._plan_calculation(task_text)
        if intent == TaskIntent.FILE_READ:
            return self._plan_file_read(task_text)
        if intent == TaskIntent.KNOWLEDGE_QUERY:
            return self._plan_knowledge_query(task_text)
        return self._plan_general(task_text)

    # -- individual planners -------------------------------------------------

    def _plan_csv_analysis(self, task_text: str) -> Plan:
        filename_match = _CSV_FILENAME_RE.search(task_text)
        column_match = _AVERAGE_COLUMN_RE.search(task_text.lower())

        if not filename_match:
            return Plan(
                intent=TaskIntent.CSV_OR_DATA_ANALYSIS,
                steps=[],
                reasoning="Task mentions CSV/data analysis but no filename could be found in the text.",
            )

        filename = filename_match.group(1).strip().rstrip(".,?!")
        column_hint = column_match.group(1).strip() if column_match else None

        code = _build_csv_analysis_code(filename, column_hint)
        step = PlanStep(
            tool_name="python_executor",
            tool_input={"code": code},
            reason=f"Load {filename} with pandas and compute the requested statistic.",
        )
        return Plan(
            intent=TaskIntent.CSV_OR_DATA_ANALYSIS,
            steps=[step],
            reasoning=f"Detected CSV file '{filename}'"
            + (f" and target column '{column_hint}'." if column_hint else "; no specific column detected."),
        )

    def _plan_calculation(self, task_text: str) -> Plan:
        expression = _extract_expression(task_text)
        step = PlanStep(
            tool_name="calculator",
            tool_input={"expression": expression},
            reason="Task looks like an arithmetic calculation.",
        )
        return Plan(
            intent=TaskIntent.CALCULATION,
            steps=[step],
            reasoning="Routed to the calculator tool for safe arithmetic evaluation.",
        )

    def _plan_file_read(self, task_text: str) -> Plan:
        match = _FILENAME_RE.search(task_text)
        filename = match.group(1).strip().rstrip(".,?!") if match else ""
        step = PlanStep(
            tool_name="file_inspector",
            tool_input={"filename": filename},
            reason="Task references a specific uploaded file.",
        )
        return Plan(
            intent=TaskIntent.FILE_READ,
            steps=[step],
            reasoning=f"Detected filename '{filename}' in the task text.",
        )

    def _plan_knowledge_query(self, task_text: str) -> Plan:
        step = PlanStep(
            tool_name="knowledge_search",
            tool_input={"query": task_text},
            reason="Task references the knowledge base / uploaded documents.",
        )
        return Plan(
            intent=TaskIntent.KNOWLEDGE_QUERY,
            steps=[step],
            reasoning="Routed to knowledge_search (RAG) to ground the answer in uploaded documents.",
        )

    def _plan_general(self, task_text: str) -> Plan:
        # Previously this always called knowledge_search as a catch-all,
        # even for queries with nothing to do with the knowledge base
        # (e.g. "write me a notification popup in Python"). That pulled in
        # irrelevant document chunks, produced confusing/wrong-looking
        # answers, and made the final-answer prompt bigger than it needed
        # to be for no benefit. Now: no specific intent matched -> just
        # answer directly with the LLM, no tool call. Explicit knowledge-
        # base references (see KNOWLEDGE_KEYWORDS / TaskIntent.KNOWLEDGE_QUERY
        # in TaskAnalyzer.analyze) still route to knowledge_search.
        return Plan(
            intent=TaskIntent.GENERAL,
            steps=[],
            reasoning="No calculation, file, CSV, or knowledge-base pattern matched. "
            "Answering directly from the LLM's own knowledge - no tool applies here.",
        )


def _build_csv_analysis_code(filename: str, column_hint: str | None) -> str:
    """Generates a small, safe pandas snippet for the python_executor tool."""
    import os

    safe_filename = os.path.join(settings.upload_dir, filename.replace('"', ""))
    column_filter = (
        f'col_hint = "{column_hint.strip()}".lower()' if column_hint else 'col_hint = None'
    )
    return f'''
import pandas as pd

path = r"{safe_filename}"
df = pd.read_csv(path)
{column_filter}

target_col = None
if col_hint:
    for c in df.columns:
        if col_hint in c.lower():
            target_col = c
            break

if target_col is None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target_col = numeric_cols[0] if numeric_cols else None

if target_col is None:
    print("NO_NUMERIC_COLUMN_FOUND")
else:
    print(f"column={{target_col}}")
    print(f"count={{df[target_col].count()}}")
    print(f"average={{df[target_col].mean():.2f}}")
    print(f"min={{df[target_col].min()}}")
    print(f"max={{df[target_col].max()}}")
'''.strip()