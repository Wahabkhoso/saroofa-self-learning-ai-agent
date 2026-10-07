from app.agent.learning import LearningModule
from app.agent.state import EvaluationResult, Plan, PlanStep, TaskIntent, ToolCallRecord
from app.database.repositories import TaskRepository
from app.memory.long_term import LongTermMemory
from app.memory.vector_memory import VectorMemory


def test_learning_module_records_experience_with_correct_outcome(db_session):
    task_repo = TaskRepository(db_session)
    task = task_repo.create("Calculate 2+2")

    ltm = LongTermMemory(db_session, vector_memory=VectorMemory())
    learning = LearningModule(ltm)

    plan = Plan(
        intent=TaskIntent.CALCULATION,
        steps=[PlanStep(tool_name="calculator", tool_input={"expression": "2+2"}, reason="x")],
        reasoning="test",
    )
    tool_calls = [ToolCallRecord(tool_name="calculator", input={}, success=True, output=4.0)]
    evaluation = EvaluationResult(
        status="SUCCESS", score=0.9, confidence=0.7, reason="ok", lesson="addition works"
    )

    stored = learning.record_experience(
        task_id=task.id, task_text="Calculate 2+2", plan=plan, tool_calls=tool_calls, evaluation=evaluation
    )
    assert stored.outcome == "SUCCESS"
    assert stored.lesson == "addition works"


def test_positive_feedback_increases_usefulness(db_session):
    task_repo = TaskRepository(db_session)
    task = task_repo.create("task")
    ltm = LongTermMemory(db_session, vector_memory=VectorMemory())
    learning = LearningModule(ltm)

    from app.database.models import TaskStatus

    exp = ltm.store_experience(
        task_id=task.id, task_summary="task", solution="sol", outcome=TaskStatus.SUCCESS,
        lesson="lesson", confidence=0.5,
    )

    from app.database.repositories import ExperienceRepository
    repo = ExperienceRepository(db_session)
    before = repo.get(exp.id).usefulness_score
    learning.apply_user_feedback(exp.id, positive=True)
    after = repo.get(exp.id).usefulness_score
    assert after > before
