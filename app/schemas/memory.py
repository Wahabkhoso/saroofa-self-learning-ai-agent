from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ExperienceOut(BaseModel):
    id: str
    task_id: str
    task_summary: str
    solution: str
    outcome: str
    lesson: str
    confidence: float
    usefulness_score: float
    times_reused: int
    created_at: datetime

    model_config = {"from_attributes": True}


class MemoryStatsOut(BaseModel):
    total_experiences: int
    successful_experiences: int
    failed_experiences: int
    average_confidence: float
