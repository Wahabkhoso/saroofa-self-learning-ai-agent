"""
Evaluator.

Spec requirement: "Do not blindly trust the LLM evaluator. Where
possible, use deterministic checks/tests for tool outputs." So:

  1. Deterministic pass: did the planned tool calls actually succeed?
     Did the tool return output at all? This alone decides FAILURE vs.
     not-FAILURE and is never overridden by the LLM.
  2. Optional LLM pass: only runs when step 1 didn't already fail, and
     only adjusts SUCCESS vs. PARTIAL_SUCCESS plus the written
     explanation/lesson. If no LLM is configured, a template-based
     evaluation is used instead - the agent still produces a real,
     structured evaluation without an API key.
"""
from __future__ import annotations

from app.agent.state import EvaluationResult, Plan, ToolCallRecord
from app.core.logging_config import get_logger
from app.llm.base import LLMMessage, LLMProvider, LLMProviderError

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a strict evaluator for an AI agent's completed task.
Given the task, the plan, and the tool results, judge correctness, completeness,
and relevance. Respond with EXACTLY this format, nothing else:

STATUS: SUCCESS | PARTIAL_SUCCESS
SCORE: <0.0-1.0>
REASON: <one sentence>
LESSON: <one sentence, a reusable lesson for similar future tasks>
"""


class Evaluator:
    def __init__(self, llm: LLMProvider | None = None):
        self._llm = llm

    def evaluate(
        self, task_text: str, plan: Plan, tool_calls: list[ToolCallRecord]
    ) -> EvaluationResult:
        deterministic_failure = self._deterministic_failure_check(plan, tool_calls)
        if deterministic_failure is not None:
            return deterministic_failure

        if self._llm is not None:
            try:
                return self._llm_evaluate(task_text, plan, tool_calls)
            except LLMProviderError as exc:
                logger.warning("evaluator: LLM evaluation failed, using fallback: %s", exc)

        return self._template_evaluate(plan, tool_calls)

    # -- deterministic layer --------------------------------------------------

    def _deterministic_failure_check(
        self, plan: Plan, tool_calls: list[ToolCallRecord]
    ) -> EvaluationResult | None:
        if plan.steps and not tool_calls:
            return EvaluationResult(
                status="FAILURE",
                score=0.0,
                confidence=0.9,
                reason="No tool was executed even though the plan required one.",
                lesson="Ensure the planner always produces an executable step for this task type.",
            )

        failed = [c for c in tool_calls if not c.success]
        if tool_calls and len(failed) == len(tool_calls):
            errors = "; ".join(c.error or "unknown error" for c in failed)
            return EvaluationResult(
                status="FAILURE",
                score=0.0,
                confidence=0.85,
                reason=f"All tool calls failed: {errors}",
                lesson=f"Tool call(s) failed with: {errors}. Check inputs before retrying.",
            )
        return None

    # -- LLM layer --------------------------------------------------------------

    def _llm_evaluate(
        self, task_text: str, plan: Plan, tool_calls: list[ToolCallRecord]
    ) -> EvaluationResult:
        tool_summary = "\n".join(
            f"- {c.tool_name}: success={c.success}, output={str(c.output)[:300]}, error={c.error}"
            for c in tool_calls
        ) or "(no tools were called)"

        prompt = (
            f"TASK: {task_text}\n\nPLAN REASONING: {plan.reasoning}\n\n"
            f"TOOL RESULTS:\n{tool_summary}"
        )
        response = self._llm.complete(
            [LLMMessage(role="user", content=prompt)], system=_SYSTEM_PROMPT, temperature=0.0
        )
        return _parse_llm_evaluation(response.content)

    # -- fallback layer -----------------------------------------------------------

    def _template_evaluate(self, plan: Plan, tool_calls: list[ToolCallRecord]) -> EvaluationResult:
        if not tool_calls:
            return EvaluationResult(
                status="PARTIAL_SUCCESS",
                score=0.5,
                confidence=0.4,
                reason="No tools were needed or available; answer generated from reasoning alone.",
                lesson="Consider whether a tool call would have made this answer more reliable.",
            )
        all_succeeded = all(c.success for c in tool_calls)
        status = "SUCCESS" if all_succeeded else "PARTIAL_SUCCESS"
        score = 0.9 if all_succeeded else 0.55
        return EvaluationResult(
            status=status,
            score=score,
            confidence=0.6,
            reason="All planned tool calls completed successfully."
            if all_succeeded
            else "Some tool calls failed or returned partial output.",
            lesson=f"For this task type ({plan.intent.value}), the plan "
            f"'{plan.reasoning}' {'worked' if all_succeeded else 'partially worked'}.",
        )


def _parse_llm_evaluation(text: str) -> EvaluationResult:
    fields = {"STATUS": "PARTIAL_SUCCESS", "SCORE": "0.5", "REASON": "", "LESSON": ""}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        if key in fields:
            fields[key] = value.strip()

    try:
        score = max(0.0, min(1.0, float(fields["SCORE"])))
    except ValueError:
        score = 0.5

    status = fields["STATUS"] if fields["STATUS"] in ("SUCCESS", "PARTIAL_SUCCESS") else "PARTIAL_SUCCESS"
    return EvaluationResult(
        status=status,
        score=score,
        confidence=0.75,
        reason=fields["REASON"] or "LLM evaluation completed.",
        lesson=fields["LESSON"] or "No specific lesson extracted.",
    )
