from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.agent import Agent
from app.api.dependencies import get_agent, get_db, rate_limit
from app.database.models import TaskStatus
from app.database.repositories import TaskRepository
from app.llm.base import LLMImage, LLMMessage
from app.memory.short_term import session_memory_store
from app.schemas.task import ChatRequest, ChatResponse, EvaluationOut, ToolCallOut, TraceStepOut

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    agent: Agent = Depends(get_agent),
) -> ChatResponse:
    task_repo = TaskRepository(db)

    if payload.message.strip():
        display_text = payload.message.strip()
    elif payload.document_text:
        display_text = f"(document attached: {payload.document_filename or 'file'})"
    else:
        display_text = "(image attached)"

    task = task_repo.create(display_text, session_id=payload.session_id)

    image = (
        LLMImage(base64_data=payload.image_base64, mime_type=payload.image_mime_type)
        if payload.image_base64
        else None
    )

    # Short-term (session) memory: pull recent turns *before* adding this
    # message, so the agent can resolve follow-ups like "give it to me" or
    # "continue" against what was actually said earlier in this session.
    # This is in-process only (see app/memory/short_term.py) - it resets
    # if the server restarts, unlike the persistent long-term memory.
    session = session_memory_store.get_or_create(payload.session_id)
    conversation_history = [
        LLMMessage(role="assistant" if turn.role == "agent" else "user", content=turn.content)
        for turn in session.recent_history(n=8)
    ]
    session.add_user_message(display_text)

    result = agent.run(
        task_id=task.id,
        task_text=payload.message,
        conversation_history=conversation_history,
        image=image,
        document_text=payload.document_text,
        document_filename=payload.document_filename,
    )
    session.add_agent_message(result.final_answer)

    task_repo.update_result(
        task,
        status=TaskStatus(result.evaluation.status),
        result=result.final_answer,
        score=result.evaluation.score,
        confidence=result.evaluation.confidence,
        error=None if result.evaluation.status != "FAILURE" else result.evaluation.reason,
        tools_used=[c.tool_name for c in result.tool_calls],
        plan=result.plan.reasoning,
        retries=result.retries,
        trace=result.trace,
        agent_confidence=result.agent_confidence,
    )

    return ChatResponse(
        task_id=task.id,
        answer=result.final_answer,
        intent=result.plan.intent.value,
        activity_log=result.activity_log,
        trace=[TraceStepOut(**step) for step in result.trace],
        tool_calls=[
            ToolCallOut(
                tool_name=c.tool_name,
                success=c.success,
                output=c.output,
                error=c.error,
                duration_seconds=c.duration_seconds,
            )
            for c in result.tool_calls
        ],
        evaluation=EvaluationOut(
            status=result.evaluation.status,
            score=result.evaluation.score,
            confidence=result.evaluation.confidence,
            reason=result.evaluation.reason,
            lesson=result.evaluation.lesson,
        ),
        retries=result.retries,
        retrieved_experiences=result.retrieved_experiences,
        experience_id=result.experience_id,
        agent_confidence=result.agent_confidence,
        failure_category=result.failure_category,
    )