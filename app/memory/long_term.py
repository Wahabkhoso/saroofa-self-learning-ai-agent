"""
Long-term memory: persistent, survives restarts.

Two backing stores, kept in sync:
  - SQL (Experience table): the structured, authoritative record.
  - Vector memory (Chroma): enables similarity search over past tasks.

This module is the only place that writes to both, so they can't drift
apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.database.models import TaskStatus
from app.database.repositories import ExperienceRepository
from app.memory.vector_memory import RetrievedExperience, VectorMemory

# Ranking weights for Experience-Based Adaptive Memory (spec section 5).
# Kept as plain constants - deterministic and explainable, not a learned
# model. similarity dominates (it's *the* signal for "is this relevant at
# all"), the rest re-order among genuinely similar candidates.
_WEIGHT_SIMILARITY = 0.55
_WEIGHT_USEFULNESS = 0.25
_WEIGHT_RECENCY = 0.10
_WEIGHT_CONFIDENCE = 0.10
_RECENCY_HALF_LIFE_DAYS = 30.0


@dataclass
class StoredExperience:
    id: str
    task_summary: str
    lesson: str
    outcome: str
    confidence: float


class LongTermMemory:
    def __init__(self, db: Session, vector_memory: VectorMemory | None = None):
        self.db = db
        self.experience_repo = ExperienceRepository(db)
        self.vector_memory = vector_memory or VectorMemory()

    def store_experience(
        self,
        *,
        task_id: str,
        task_summary: str,
        solution: str,
        outcome: TaskStatus,
        lesson: str,
        confidence: float,
        failure_category: str | None = None,
    ) -> StoredExperience:
        exp = self.experience_repo.create(
            task_id=task_id,
            task_summary=task_summary,
            solution=solution,
            outcome=outcome,
            lesson=lesson,
            confidence=confidence,
            failure_category=failure_category,
        )
        self.vector_memory.add_experience(
            experience_id=exp.id,
            task_summary=task_summary,
            lesson=lesson,
            solution=solution,
            outcome=outcome.value,
            confidence=confidence,
        )
        return StoredExperience(
            id=exp.id,
            task_summary=task_summary,
            lesson=lesson,
            outcome=outcome.value,
            confidence=confidence,
        )

    def retrieve_similar(self, task_text: str, top_k: int = 3) -> list[RetrievedExperience]:
        # Over-fetch raw candidates so ranking has something to actually
        # re-order beyond pure vector similarity.
        raw_candidates = self.vector_memory.search(task_text, top_k=max(top_k * 4, top_k))
        ranked = self._rank(raw_candidates)[:top_k]
        for r in ranked:
            self.experience_repo.mark_reused(r.experience_id)
        return ranked

    def _rank(self, candidates: list[RetrievedExperience]) -> list[RetrievedExperience]:
        """Deterministic re-ranking: similarity + usefulness + recency +
        confidence. See module-level weight constants for the exact
        formula - this is intentionally simple and explainable, not a
        black box."""
        now = datetime.now(timezone.utc)
        for c in candidates:
            exp = self.experience_repo.get(c.experience_id)
            usefulness = exp.usefulness_score if exp else 0.5
            age_days = 0.0
            if exp is not None:
                created = exp.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now - created).total_seconds() / 86400)
            recency_score = math.exp(-age_days / _RECENCY_HALF_LIFE_DAYS)

            c.usefulness_score = usefulness
            c.rank_score = round(
                _WEIGHT_SIMILARITY * c.similarity
                + _WEIGHT_USEFULNESS * usefulness
                + _WEIGHT_RECENCY * recency_score
                + _WEIGHT_CONFIDENCE * c.confidence,
                4,
            )
        return sorted(candidates, key=lambda c: c.rank_score, reverse=True)

    def apply_feedback(self, experience_id: str, *, positive: bool) -> None:
        delta = 0.1 if positive else -0.15
        self.experience_repo.adjust_usefulness(experience_id, delta)

    def stats(self) -> dict:
        return self.experience_repo.stats()