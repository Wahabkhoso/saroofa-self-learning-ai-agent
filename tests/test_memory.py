from app.database.models import TaskStatus
from app.database.repositories import TaskRepository
from app.memory.long_term import LongTermMemory
from app.memory.vector_memory import VectorMemory


def test_store_and_retrieve_similar_experience(db_session):
    task_repo = TaskRepository(db_session)
    task = task_repo.create("How can I fix a Python CSV PermissionError?")

    ltm = LongTermMemory(db_session, vector_memory=VectorMemory())
    ltm.store_experience(
        task_id=task.id,
        task_summary="How can I fix a Python CSV PermissionError?",
        solution="Close the file in Excel before reading it with pandas.",
        outcome=TaskStatus.SUCCESS,
        lesson="A PermissionError on a CSV usually means it's open in another program.",
        confidence=0.8,
    )

    # Different wording, same underlying problem - this is the RAG-memory
    # scenario from the project spec.
    results = ltm.retrieve_similar("I'm getting permission denied when saving students.csv", top_k=3)
    assert len(results) >= 1
    assert "permission" in results[0].lesson.lower()


def test_retrieve_similar_on_empty_memory_returns_empty(db_session):
    ltm = LongTermMemory(db_session, vector_memory=VectorMemory())
    results = ltm.retrieve_similar("anything at all")
    assert results == []


def test_experience_stats(db_session):
    task_repo = TaskRepository(db_session)
    ltm = LongTermMemory(db_session, vector_memory=VectorMemory())

    t1 = task_repo.create("task one")
    t2 = task_repo.create("task two")
    ltm.store_experience(
        task_id=t1.id, task_summary="task one", solution="did it", outcome=TaskStatus.SUCCESS,
        lesson="worked", confidence=0.9,
    )
    ltm.store_experience(
        task_id=t2.id, task_summary="task two", solution="failed", outcome=TaskStatus.FAILURE,
        lesson="didn't work", confidence=0.3,
    )

    stats = ltm.stats()
    assert stats["total_experiences"] == 2
    assert stats["successful_experiences"] == 1
    assert stats["failed_experiences"] == 1


def test_apply_feedback_adjusts_usefulness(db_session):
    task_repo = TaskRepository(db_session)
    ltm = LongTermMemory(db_session, vector_memory=VectorMemory())
    task = task_repo.create("task")
    exp = ltm.store_experience(
        task_id=task.id, task_summary="task", solution="sol", outcome=TaskStatus.SUCCESS,
        lesson="lesson", confidence=0.5,
    )

    from app.database.repositories import ExperienceRepository
    repo = ExperienceRepository(db_session)
    before = repo.get(exp.id).usefulness_score

    ltm.apply_feedback(exp.id, positive=True)
    after_positive = repo.get(exp.id).usefulness_score
    assert after_positive > before

    ltm.apply_feedback(exp.id, positive=False)
    after_negative = repo.get(exp.id).usefulness_score
    assert after_negative < after_positive
