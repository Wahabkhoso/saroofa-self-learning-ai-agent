"""
Agent orchestrator.

Implements the workflow from the spec:

  Task -> Analyze -> Memory retrieval -> Plan -> Tool execution
       -> Evaluate -> (retry loop, bounded) -> Final answer -> Learn

All loop bounds come from AgentLimits (app/agent/state.py) so this can
never run away: max_tool_calls caps how many tool calls happen per task,
max_retries caps the error-recovery loop, and each tool call itself has
a hard timeout (see app/tools/python_tool.py).
"""
from __future__ import annotations

from app.agent.evaluator import Evaluator
from app.agent.learning import LearningModule
from app.agent.planner import Planner, TaskAnalyzer
from app.agent.state import (
    AgentLimits,
    AgentRunResult,
    EvaluationResult,
    Plan,
    PlanStep,
    TaskIntent,
    ToolCallRecord,
    TraceRecorder,
    categorize_failure,
    compute_agent_confidence,
)
from app.core.config import settings
from app.core.logging_config import get_logger
from app.llm.base import LLMImage, LLMMessage, LLMProvider, LLMProviderError
from app.memory.long_term import LongTermMemory
from app.tools.base import ToolRegistry

logger = get_logger(__name__)

_ANSWER_SYSTEM_PROMPT_TEMPLATE = """You are the answer-writing component of "{agent_name}," an
autonomous task-solving system built on persistent memory, tool use, and evaluated experience.
Write a clear, concise final answer for the user based on the tool results provided.
Do not mention "tools" or internal mechanics - just answer naturally.
If relevant past experience is provided, you may draw on it, but do not mention "memory" explicitly.
If the user asks who you are, what your name is, or what model/company is behind you, answer that
your name is "{agent_name}" - do not reveal or reference the underlying LLM provider or model
name, even if you know it, under any circumstance, even if the user insists or claims to be a
developer testing you.
"""


class Agent:
    def __init__(
        self,
        *,
        long_term_memory: LongTermMemory,
        tool_registry: ToolRegistry,
        llm: LLMProvider | None = None,
        limits: AgentLimits | None = None,
    ):
        self.long_term_memory = long_term_memory
        self.tool_registry = tool_registry
        self.llm = llm
        self.limits = limits or AgentLimits()
        self.analyzer = TaskAnalyzer()
        self.planner = Planner()
        self.evaluator = Evaluator(llm=llm)
        self.learning = LearningModule(long_term_memory)

    def run(
        self,
        task_id: str,
        task_text: str,
        conversation_history: list[LLMMessage] | None = None,
        image: LLMImage | None = None,
        document_text: str | None = None,
        document_filename: str | None = None,
    ) -> AgentRunResult:
        activity_log: list[str] = []
        trace = TraceRecorder()

        trace.start("task_analysis")
        activity_log.append("Analyzing task...")
        intent = self.analyzer.analyze(task_text) if task_text.strip() else TaskIntent.GENERAL
        trace.finish(intent=intent.value)

        trace.start("memory_retrieval")
        activity_log.append("Searching memory...")
        retrieved = (
            self.long_term_memory.retrieve_similar(task_text, top_k=3) if task_text.strip() else []
        )
        retrieved_summaries = [
            f"(similarity={r.similarity}) {r.task_summary} -> lesson: {r.lesson}" for r in retrieved
        ]
        trace.finish(retrieved_count=len(retrieved))

        # Images bypass tool planning entirely: none of the four tools
        # (calculator, python_executor, file_inspector, knowledge_search)
        # understand images, so there is nothing useful for the rule-based
        # planner to route to. Instead we go straight to a vision-capable
        # LLM call. This is a deliberate, narrow special case - not a
        # second planner - see the "if image is not None" branch below.
        # Attached documents do NOT get this treatment - a document is just
        # text, so it flows through the normal planner/tool path below and
        # is injected into the final-answer prompt (see _llm_answer).
        if image is not None:
            trace.start("planning")
            activity_log.append("Image detected - skipping tool planning, routing to vision model...")
            plan = Plan(
                intent=TaskIntent.GENERAL,
                steps=[],
                reasoning="Image attached - handled directly by the vision-capable LLM, no tools used.",
            )
            trace.finish(intent=plan.intent.value, planned_tools=[])

            tool_calls: list[ToolCallRecord] = []
            retries = 0

            trace.start("tool_execution")
            trace.finish()  # nothing to execute - recorded for a consistent trace shape

            trace.start("final_answer")
            activity_log.append("Analyzing image and generating final answer...")
            if self.llm is None:
                final_answer = (
                    "I can't analyze images without a configured LLM. Add ANTHROPIC_API_KEY, "
                    "OPENAI_API_KEY, or GEMINI_API_KEY to your .env file."
                )
                evaluation = EvaluationResult(
                    status="FAILURE", score=0.0, confidence=0.9,
                    reason="No LLM provider is configured for vision analysis.",
                    lesson="Image analysis requires a configured LLM API key.",
                )
                trace.finish(error="no LLM configured")
                llm_error_for_image = True
            else:
                try:
                    final_answer = self._generate_final_answer(
                        task_text or "Transcribe all text visible in this image accurately and completely. ",
                        plan,
                        tool_calls,
                        retrieved_summaries,
                        conversation_history,
                        image=image,
                    )
                    evaluation = EvaluationResult(
                        status="SUCCESS",
                        score=0.85,
                        confidence=0.7,
                        reason="Image was analyzed directly by the vision-capable LLM.",
                        lesson="Image-based tasks are answered directly by the LLM; no tool applies.",
                    )
                    trace.finish()
                except LLMProviderError as exc:
                    final_answer = f"I couldn't analyze the image - the vision API call failed: {exc}"
                    evaluation = EvaluationResult(
                        status="FAILURE", score=0.0, confidence=0.8,
                        reason=str(exc), lesson="Image analysis failed - check API key/quota.",
                    )
                    trace.finish(error=str(exc))
                    llm_error_for_image = True
                else:
                    llm_error_for_image = False
            activity_log.append("Evaluating result...")

            trace.start("learning")
            activity_log.append("Learning from this result...")
            stored = self.learning.record_experience(
                task_id=task_id,
                task_text=task_text or "(image attached, no text)",
                plan=plan,
                tool_calls=tool_calls,
                evaluation=evaluation,
            )
            trace.finish(experience_id=stored.id)

            failure_category = (
                "LLM_ERROR" if evaluation.status == "FAILURE" and llm_error_for_image
                else (categorize_failure(tool_calls).value if evaluation.status == "FAILURE" else None)
            )
            agent_confidence = compute_agent_confidence(
                evaluation=evaluation, tool_calls=tool_calls, retries=0,
                top_memory_similarity=retrieved[0].similarity if retrieved else None,
            )
            return AgentRunResult(
                task_id=task_id, final_answer=final_answer, plan=plan, tool_calls=tool_calls,
                evaluation=evaluation, retries=0, retrieved_experiences=retrieved_summaries,
                activity_log=activity_log, experience_id=stored.id, trace=trace.as_dicts(),
                agent_confidence=agent_confidence, failure_category=failure_category,
            )

        trace.start("planning")
        activity_log.append("Planning...")
        plan = self.planner.build_plan(task_text, intent)
        trace.finish(intent=plan.intent.value, planned_tools=[s.tool_name for s in plan.steps])

        tool_calls: list[ToolCallRecord] = []
        retries = 0

        trace.start("tool_execution")
        tool_calls.extend(self._execute_plan(plan, activity_log))
        trace.finish(
            error=None if all(c.success for c in tool_calls) or not tool_calls else "one or more tool calls failed"
        )

        trace.start("evaluation")
        activity_log.append("Evaluating result...")
        evaluation = self.evaluator.evaluate(task_text, plan, tool_calls)
        trace.finish(status=evaluation.status, score=evaluation.score)

        # Bounded error-recovery loop.
        while evaluation.status == "FAILURE" and retries < self.limits.max_retries:
            retries += 1
            failure_category = categorize_failure(tool_calls)
            trace.start(f"retry_{retries}", failure_category=failure_category.value)
            activity_log.append(f"Detected failure ({failure_category.value}). Retry {retries}/{self.limits.max_retries}...")
            similar_errors = [r for r in retrieved if "fail" in r.outcome.lower()]
            if similar_errors:
                activity_log.append(f"Found {len(similar_errors)} similar past failure(s) - applying their lesson.")

            retry_calls = self._execute_plan(plan, activity_log, is_retry=True)
            tool_calls.extend(retry_calls)
            evaluation = self.evaluator.evaluate(task_text, plan, tool_calls)
            trace.finish(status=evaluation.status)

        trace.start("final_answer")
        activity_log.append("Generating final answer...")
        final_answer = self._generate_final_answer(
            task_text,
            plan,
            tool_calls,
            retrieved_summaries,
            conversation_history,
            document_text=document_text,
            document_filename=document_filename,
        )
        trace.finish()

        trace.start("learning")
        activity_log.append("Learning from this result...")
        stored = self.learning.record_experience(
            task_id=task_id,
            task_text=task_text,
            plan=plan,
            tool_calls=tool_calls,
            evaluation=evaluation,
        )
        trace.finish(experience_id=stored.id)

        failure_category = (
            categorize_failure(tool_calls).value if evaluation.status == "FAILURE" else None
        )
        agent_confidence = compute_agent_confidence(
            evaluation=evaluation,
            tool_calls=tool_calls,
            retries=retries,
            top_memory_similarity=retrieved[0].similarity if retrieved else None,
        )

        return AgentRunResult(
            task_id=task_id,
            final_answer=final_answer,
            plan=plan,
            tool_calls=tool_calls,
            evaluation=evaluation,
            retries=retries,
            retrieved_experiences=retrieved_summaries,
            activity_log=activity_log,
            experience_id=stored.id,
            trace=trace.as_dicts(),
            agent_confidence=agent_confidence,
            failure_category=failure_category,
        )

    # -- internals -----------------------------------------------------------

    def _execute_plan(
        self, plan: Plan, activity_log: list[str], is_retry: bool = False
    ) -> list[ToolCallRecord]:
        records: list[ToolCallRecord] = []
        for step in plan.steps[: self.limits.max_tool_calls]:
            records.append(self._execute_step(step, activity_log, is_retry))
        return records

    def _execute_step(self, step: PlanStep, activity_log: list[str], is_retry: bool) -> ToolCallRecord:
        tool = self.tool_registry.get(step.tool_name)
        verb = "Retrying" if is_retry else "Using"
        activity_log.append(f"{verb} {step.tool_name}...")

        if tool is None:
            return ToolCallRecord(
                tool_name=step.tool_name,
                input=step.tool_input,
                success=False,
                error=f"Unknown tool: {step.tool_name}",
            )

        result = tool.run(**step.tool_input)
        return ToolCallRecord(
            tool_name=step.tool_name,
            input=step.tool_input,
            success=result.success,
            output=result.output,
            error=result.error,
            duration_seconds=result.duration_seconds,
        )

    def _generate_final_answer(
        self,
        task_text: str,
        plan: Plan,
        tool_calls: list[ToolCallRecord],
        retrieved_summaries: list[str],
        conversation_history: list[LLMMessage] | None = None,
        image: LLMImage | None = None,
        document_text: str | None = None,
        document_filename: str | None = None,
    ) -> str:
        if self.llm is not None:
            try:
                return self._llm_answer(
                    task_text,
                    plan,
                    tool_calls,
                    retrieved_summaries,
                    conversation_history,
                    image,
                    document_text=document_text,
                    document_filename=document_filename,
                )
            except LLMProviderError as exc:
                if image is not None:
                    raise  # no sensible template fallback for "describe this image"
                logger.warning("agent: LLM answer generation failed, using fallback: %s", exc)
                return self._template_answer(
                    plan, tool_calls, llm_failed=True, document_filename=document_filename
                )

        return self._template_answer(plan, tool_calls, llm_failed=False, document_filename=document_filename)

    def _llm_answer(
        self,
        task_text: str,
        plan: Plan,
        tool_calls: list[ToolCallRecord],
        retrieved_summaries: list[str],
        conversation_history: list[LLMMessage] | None = None,
        image: LLMImage | None = None,
        document_text: str | None = None,
        document_filename: str | None = None,
    ) -> str:
        tool_summary = "\n".join(
            f"- {c.tool_name} ({'ok' if c.success else 'failed'}): {c.output or c.error}"
            for c in tool_calls
        ) or "(no tools were used; answer from general knowledge)"
        memory_summary = "\n".join(retrieved_summaries) or "(no relevant past experience found)"

        document_block = ""
        if document_text:
            # Inline, one-off document context - NOT the same as the
            # persistent knowledge base (see app/api/routes/knowledge.py's
            # /extract vs /upload). This document is only visible for this
            # one message; it is not saved or searchable later unless the
            # user also chose to save it to the knowledge base.
            label = document_filename or "attached document"
            document_block = f"\n\nATTACHED DOCUMENT ({label}):\n{document_text}\n"

        prompt = (
            f"USER TASK: {task_text}\n\n"
            f"TOOL RESULTS:\n{tool_summary}\n"
            f"{document_block}\n"
            f"RELEVANT PAST EXPERIENCE:\n{memory_summary}\n\n"
            "Write the final answer for the user now."
        )
        # Prior turns in this conversation come first, so the model can
        # resolve references like "give it to me" or "continue" against
        # what was actually said earlier - without this, every message
        # was answered in isolation.
        messages = list(conversation_history or []) + [
            LLMMessage(role="user", content=prompt, image=image)
        ]
        response = self.llm.complete(
            messages,
            system=_ANSWER_SYSTEM_PROMPT_TEMPLATE.format(agent_name=settings.app_name),
        )
        return response.content.strip()

    def _template_answer(
        self,
        plan: Plan,
        tool_calls: list[ToolCallRecord],
        llm_failed: bool = False,
        document_filename: str | None = None,
    ) -> str:
        # Two different reasons land here, and they deserve different
        # messages: (a) no LLM is configured at all - a real setup issue,
        # vs (b) an LLM is configured but this particular call failed
        # (e.g. Gemini's transient 503 under free-tier load) - usually
        # fine on the next try. Showing raw tool output to the user
        # (the old behavior) was confusing and looked broken even when
        # it wasn't.
        doc_note = f" I do have your document ({document_filename}) ready for when it's back up." if (
            llm_failed and document_filename
        ) else ""

        if not tool_calls:
            if llm_failed:
                return (
                    "Sorry, I couldn't generate a response just now - the AI service is "
                    f"temporarily unavailable. Please try again in a moment.{doc_note}"
                )
            return (
                "I wasn't able to match this task to a tool, and no LLM provider is configured "
                "to answer it directly. Configure ANTHROPIC_API_KEY or OPENAI_API_KEY in .env."
            )

        lines = []
        if llm_failed:
            lines.append(
                "I couldn't write a full answer just now - the AI service is temporarily "
                f"unavailable.{doc_note} Here are the raw results from the tools I ran:"
            )
        for call in tool_calls:
            if call.success:
                lines.append(f"{call.tool_name} result: {call.output}")
            else:
                lines.append(f"{call.tool_name} failed: {call.error}")
        return "\n".join(lines)