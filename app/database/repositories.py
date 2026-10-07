"""
Repositories: thin data-access classes around each table.

Keeping these separate from the agent logic means the agent never
touches SQLAlchemy directly, which makes both sides easier to test.
"""
from __future__ import annotations

import json
from typing import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.database.models import Document, Experience, Feedback, FeedbackType, Task, TaskStatus


class TaskRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, user_input: str, session_id: str = "default") -> Task:
        task = Task(user_input=user_input, session_id=session_id, status=TaskStatus.PENDING)
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task

    def get(self, task_id: str) -> Task | None:
        return self.db.get(Task, task_id)

    def list_recent(self, limit: int = 50, session_id: str | None = None) -> Sequence[Task]:
        stmt = select(Task)
        if session_id is not None:
            stmt = stmt.where(Task.session_id == session_id)
        stmt = stmt.order_by(Task.created_at.desc()).limit(limit)
        return self.db.execute(stmt).scalars().all()

    def list_sessions(self, limit: int = 30) -> list[dict]:
        """One row per distinct session_id, newest activity first.
        NOTE: this runs one extra query per session to fetch its first
        message as the title (N+1) - fine at demo scale, would need a
        proper subquery or a sessions table if this ever needs to scale."""
        counts_stmt = (
            select(
                Task.session_id,
                func.count(Task.id).label("task_count"),
                func.max(Task.created_at).label("last_activity"),
            )
            .group_by(Task.session_id)
            .order_by(func.max(Task.created_at).desc())
            .limit(limit)
        )
        rows = self.db.execute(counts_stmt).all()

        sessions = []
        for session_id, task_count, last_activity in rows:
            first_input = self.db.execute(
                select(Task.user_input)
                .where(Task.session_id == session_id)
                .order_by(Task.created_at.asc())
                .limit(1)
            ).scalar_one_or_none()
            sessions.append({
                "session_id": session_id,
                "title": (first_input or "New chat")[:60],
                "task_count": task_count,
                "last_activity": last_activity,
            })
        return sessions

    def delete_session(self, session_id: str) -> int:
        """Delete every Task in this session, plus its Feedback and
        Experience rows (there's no cascade-delete configured on the
        foreign keys, and SQLite's FK enforcement is off by default in
        this project - deleting Task rows alone would silently leave
        orphaned Feedback/Experience rows behind). Child tables first,
        then the parent. Returns the number of tasks deleted."""
        task_ids_stmt = select(Task.id).where(Task.session_id == session_id)
        task_ids = [row[0] for row in self.db.execute(task_ids_stmt).all()]
        if not task_ids:
            return 0

        self.db.execute(delete(Feedback).where(Feedback.task_id.in_(task_ids)))
        self.db.execute(delete(Experience).where(Experience.task_id.in_(task_ids)))
        result = self.db.execute(delete(Task).where(Task.session_id == session_id))
        self.db.commit()
        return result.rowcount

    def update_result(
        self,
        task: Task,
        *,
        status: TaskStatus,
        result: str | None,
        score: float | None,
        confidence: float | None,
        error: str | None,
        tools_used: list[str] | None,
        plan: str | None,
        retries: int,
        trace: list[dict] | None = None,
        agent_confidence: float | None = None,
    ) -> Task:
        task.status = status
        task.result = result
        task.score = score
        task.confidence = confidence
        task.error = error
        task.tools_used = json.dumps(tools_used or [])
        task.plan = plan
        task.retries = retries
        task.trace = json.dumps(trace or [])
        task.agent_confidence = agent_confidence
        self.db.commit()
        self.db.refresh(task)
        return task

    def get_trace(self, task_id: str) -> list[dict] | None:
        task = self.get(task_id)
        if task is None or not task.trace:
            return None
        return json.loads(task.trace)


class ExperienceRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        *,
        task_id: str,
        task_summary: str,
        solution: str,
        outcome: TaskStatus,
        lesson: str,
        confidence: float,
        failure_category: str | None = None,
    ) -> Experience:
        exp = Experience(
            task_id=task_id,
            task_summary=task_summary,
            solution=solution,
            outcome=outcome,
            lesson=lesson,
            confidence=confidence,
            failure_category=failure_category,
        )
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def get(self, experience_id: str) -> Experience | None:
        return self.db.get(Experience, experience_id)

    def list_recent(self, limit: int = 50) -> Sequence[Experience]:
        stmt = select(Experience).order_by(Experience.created_at.desc()).limit(limit)
        return self.db.execute(stmt).scalars().all()

    def adjust_usefulness(self, experience_id: str, delta: float) -> Experience | None:
        exp = self.get(experience_id)
        if exp is None:
            return None
        exp.usefulness_score = max(0.0, min(1.0, exp.usefulness_score + delta))
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def mark_reused(self, experience_id: str) -> None:
        exp = self.get(experience_id)
        if exp is not None:
            exp.times_reused += 1
            self.db.commit()

    def stats(self) -> dict:
        total = self.db.execute(select(func.count(Experience.id))).scalar_one()
        successes = self.db.execute(
            select(func.count(Experience.id)).where(Experience.outcome == TaskStatus.SUCCESS)
        ).scalar_one()
        failures = self.db.execute(
            select(func.count(Experience.id)).where(Experience.outcome == TaskStatus.FAILURE)
        ).scalar_one()
        avg_conf = self.db.execute(select(func.avg(Experience.confidence))).scalar_one()
        return {
            "total_experiences": total,
            "successful_experiences": successes,
            "failed_experiences": failures,
            "average_confidence": round(avg_conf, 3) if avg_conf is not None else 0.0,
        }


class FeedbackRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self, *, task_id: str, feedback_type: FeedbackType, feedback_text: str | None
    ) -> Feedback:
        fb = Feedback(task_id=task_id, feedback_type=feedback_type, feedback_text=feedback_text)
        self.db.add(fb)
        self.db.commit()
        self.db.refresh(fb)
        return fb

    def list_for_task(self, task_id: str) -> Sequence[Feedback]:
        stmt = select(Feedback).where(Feedback.task_id == task_id)
        return self.db.execute(stmt).scalars().all()


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, *, filename: str, file_type: str, size_bytes: int, chunk_count: int) -> Document:
        doc = Document(
            filename=filename, file_type=file_type, size_bytes=size_bytes, chunk_count=chunk_count
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get(self, document_id: str) -> Document | None:
        return self.db.get(Document, document_id)

    def list_all(self) -> Sequence[Document]:
        stmt = select(Document).order_by(Document.created_at.desc())
        return self.db.execute(stmt).scalars().all()

    def delete(self, document_id: str) -> bool:
        doc = self.get(document_id)
        if doc is None:
            return False
        self.db.delete(doc)
        self.db.commit()
        return True