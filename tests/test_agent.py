from app.agent.agent import Agent
from app.agent.state import AgentLimits
from app.database.repositories import ExperienceRepository
from app.memory.long_term import LongTermMemory
from app.memory.vector_memory import VectorMemory
from app.tools.registry import build_default_registry


def _make_agent(db_session, limits=None):
    ltm = LongTermMemory(db_session, vector_memory=VectorMemory())
    registry = build_default_registry()
    return Agent(long_term_memory=ltm, tool_registry=registry, llm=None, limits=limits or AgentLimits())


def test_agent_runs_calculation_task_successfully(db_session):
    agent = _make_agent(db_session)
    result = agent.run(task_id="t1", task_text="Calculate (12 + 8) * 3 / sqrt(4)")

    assert result.evaluation.status == "SUCCESS"
    assert result.tool_calls[0].tool_name == "calculator"
    assert result.tool_calls[0].output == 30.0
    assert result.retries == 0


def test_agent_stores_experience_after_run(db_session):
    agent = _make_agent(db_session)
    agent.run(task_id="t1", task_text="Calculate 2 + 2")

    exp_repo = ExperienceRepository(db_session)
    experiences = exp_repo.list_recent()
    assert len(experiences) == 1
    assert experiences[0].outcome.value == "SUCCESS"


def test_agent_retrieves_prior_experience_on_similar_task(db_session):
    agent = _make_agent(db_session)
    agent.run(task_id="t1", task_text="Calculate 100 / 4")
    result = agent.run(task_id="t2", task_text="Please calculate 100 / 4 again")

    assert len(result.retrieved_experiences) >= 1


def test_agent_bounded_retries_on_repeated_failure(db_session):
    limits = AgentLimits(max_retries=2, max_tool_calls=8, max_iterations=6, tool_timeout_seconds=5)
    agent = _make_agent(db_session, limits=limits)

    # Nonexistent CSV file -> the python_executor will fail every time.
    result = agent.run(task_id="t1", task_text="Analyze does_not_exist_at_all.csv and calculate the average bonus")

    assert result.evaluation.status == "FAILURE"
    assert result.retries == 2  # never exceeds max_retries
    # Should have tried: 1 initial + 2 retries = 3 tool calls total.
    assert len(result.tool_calls) == 3


def test_agent_csv_analysis_end_to_end(db_session, tmp_path):
    from app.core.config import settings

    settings.upload_dir = str(tmp_path)
    csv_path = tmp_path / "students.csv"
    csv_path.write_text("name,salary\nAlice,85000\nBob,62000\nCarol,91000\n")

    agent = _make_agent(db_session)
    result = agent.run(
        task_id="t1", task_text="I need to analyze students.csv and calculate the average salary"
    )

    assert result.evaluation.status == "SUCCESS"
    assert "79333.33" in result.final_answer
