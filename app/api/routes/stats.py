from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_long_term_memory
from app.core.config import settings
from app.database.models import Task, TaskStatus
from app.memory.long_term import LongTermMemory

router = APIRouter(tags=["stats"])


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app_name": settings.app_name, "llm_provider": settings.llm_provider}


@router.get("/api/stats")
def stats(
    db: Session = Depends(get_db),
    long_term_memory: LongTermMemory = Depends(get_long_term_memory),
) -> dict:
    total_tasks = db.execute(select(func.count(Task.id))).scalar_one()
    success_tasks = db.execute(
        select(func.count(Task.id)).where(Task.status == TaskStatus.SUCCESS)
    ).scalar_one()
    failed_tasks = db.execute(
        select(func.count(Task.id)).where(Task.status == TaskStatus.FAILURE)
    ).scalar_one()

    return {
        "total_tasks": total_tasks,
        "successful_tasks": success_tasks,
        "failed_tasks": failed_tasks,
        "memory": long_term_memory.stats(),
        "llm_provider": settings.llm_provider,
        "model_name": settings.model_name,
    }
