from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.database.repositories import ExperienceRepository
from app.schemas.memory import ExperienceOut, MemoryStatsOut

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("", response_model=list[ExperienceOut])
def list_experiences(limit: int = 50, db: Session = Depends(get_db)) -> list[ExperienceOut]:
    repo = ExperienceRepository(db)
    return [ExperienceOut.model_validate(e) for e in repo.list_recent(limit=limit)]


@router.get("/{experience_id}", response_model=ExperienceOut)
def get_experience(experience_id: str, db: Session = Depends(get_db)) -> ExperienceOut:
    repo = ExperienceRepository(db)
    exp = repo.get(experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    return ExperienceOut.model_validate(exp)


@router.delete("/{experience_id}")
def delete_experience(experience_id: str, db: Session = Depends(get_db)) -> dict:
    repo = ExperienceRepository(db)
    exp = repo.get(experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    db.delete(exp)
    db.commit()
    return {"deleted": True, "id": experience_id}
