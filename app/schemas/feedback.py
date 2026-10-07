from __future__ import annotations

from pydantic import BaseModel, Field

from app.database.models import FeedbackType


class FeedbackRequest(BaseModel):
    task_id: str
    feedback_type: FeedbackType
    feedback_text: str | None = Field(default=None, max_length=2000)
    experience_id: str | None = None


class FeedbackOut(BaseModel):
    id: str
    task_id: str
    feedback_type: FeedbackType
    feedback_text: str | None

    model_config = {"from_attributes": True}
