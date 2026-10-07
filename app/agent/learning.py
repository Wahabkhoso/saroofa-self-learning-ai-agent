"""
Learning module.

This is where "genuine feedback-based learning" (per the project brief)
actually happens, and it is important to be precise about what that
means here: the underlying LLM's weights are never touched. What
persists and improves is:

  1. A structured record of what was tried and what happened
     (app/database/models.py: Experience).
  2. A short, reusable natural-language "lesson" for that task type.
  3. A confidence/usefulness score that moves up or down based on
     explicit user feedback (see apply_feedback below).
  4. Semantic indexing (vector_memory) so a *conceptually* similar
     future task retrieves this experience even with different wording.

That combination is retrieval-augmented behavior shaping: future
answers are better because relevant past lessons are pulled into the
prompt, not because any weights changed.
"""
from __future__ import annotations

from app.agent.state import EvaluationResult, Plan, ToolCallRecord, categorize_failure
from app.database.models import TaskStatus
from app.memory.long_term import LongTermMemory, StoredExperience


class LearningModule:
    def __init__(self, long_term_memory: LongTermMemory):
        self.long_term_memory = long_term_memory

    def record_experience(
        self,
        *,
        task_id: str,
        task_text: str,
        plan: Plan,
        tool_calls: list[ToolCallRecord],
        evaluation: EvaluationResult,
    ) -> StoredExperience:
        solution = self._summarize_solution(plan, tool_calls)
        outcome = TaskStatus(evaluation.status)
        failure_category = (
            categorize_failure(tool_calls).value if outcome == TaskStatus.FAILURE else None
        )
        return self.long_term_memory.store_experience(
            task_id=task_id,
            task_summary=task_text,
            solution=solution,
            outcome=outcome,
            lesson=evaluation.lesson,
            confidence=evaluation.confidence,
            failure_category=failure_category,
        )

    def apply_user_feedback(self, experience_id: str, *, positive: bool) -> None:
        self.long_term_memory.apply_feedback(experience_id, positive=positive)

    @staticmethod
    def _summarize_solution(plan: Plan, tool_calls: list[ToolCallRecord]) -> str:
        if not tool_calls:
            return f"Answered directly using reasoning only (intent={plan.intent.value})."
        parts = []
        for call in tool_calls:
            outcome = "succeeded" if call.success else f"failed ({call.error})"
            parts.append(f"{call.tool_name} {outcome}")
        return f"Plan: {plan.reasoning} | Steps: " + "; ".join(parts)