from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.learning import LearningModule
from app.api.dependencies import get_db, get_long_term_memory
from app.database.models import FeedbackType
from app.database.repositories import FeedbackRepository, TaskRepository
from app.memory.long_term import LongTermMemory
from app.schemas.feedback import FeedbackOut, FeedbackRequest

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

_POSITIVE_TYPES = {FeedbackType.THUMBS_UP, FeedbackType.CORRECT}
_NEGATIVE_TYPES = {FeedbackType.THUMBS_DOWN, FeedbackType.INCORRECT}


@router.post("", response_model=FeedbackOut)
def submit_feedback(
    payload: FeedbackRequest,
    db: Session = Depends(get_db),
    long_term_memory: LongTermMemory = Depends(get_long_term_memory),
) -> FeedbackOut:
    task_repo = TaskRepository(db)
    task = task_repo.get(payload.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    feedback_repo = FeedbackRepository(db)
    feedback = feedback_repo.create(
        task_id=payload.task_id,
        feedback_type=payload.feedback_type,
        feedback_text=payload.feedback_text,
    )

    if payload.experience_id:
        learning = LearningModule(long_term_memory)
        if payload.feedback_type in _POSITIVE_TYPES:
            learning.apply_user_feedback(payload.experience_id, positive=True)
        elif payload.feedback_type in _NEGATIVE_TYPES:
            learning.apply_user_feedback(payload.experience_id, positive=False)

    return FeedbackOut.model_validate(feedback)
