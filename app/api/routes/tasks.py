from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.database.repositories import TaskRepository
from app.schemas.task import SessionOut, TaskOut, TraceStepOut

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(limit: int = 30, db: Session = Depends(get_db)) -> list[SessionOut]:
    repo = TaskRepository(db)
    return [SessionOut(**s) for s in repo.list_sessions(limit=limit)]


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    repo = TaskRepository(db)
    deleted_count = repo.delete_session(session_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="No chat found with that session_id")
    return {"deleted": True, "session_id": session_id, "tasks_deleted": deleted_count}


@router.get("", response_model=list[TaskOut])
def list_tasks(
    limit: int = 50,
    session_id: str | None = None,
    db: Session = Depends(get_db),
) -> list[TaskOut]:
    repo = TaskRepository(db)
    return [TaskOut.model_validate(t) for t in repo.list_recent(limit=limit, session_id=session_id)]


@router.get("/{task_id}", response_model=TaskOut)
def get_task(task_id: str, db: Session = Depends(get_db)) -> TaskOut:
    repo = TaskRepository(db)
    task = repo.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskOut.model_validate(task)


@router.get("/{task_id}/trace", response_model=list[TraceStepOut])
def get_task_trace(task_id: str, db: Session = Depends(get_db)) -> list[TraceStepOut]:
    """Structured execution trace for one task - see app/agent/state.py::TraceRecorder.
    Never contains secrets; metadata is limited to step-level bookkeeping
    (intents, tool names, counts)."""
    repo = TaskRepository(db)
    if repo.get(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    trace = repo.get_trace(task_id)
    if trace is None:
        return []
    return [TraceStepOut(**step) for step in trace]