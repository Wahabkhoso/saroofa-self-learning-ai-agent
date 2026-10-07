from app.agent.evaluator import Evaluator
from app.agent.state import Plan, PlanStep, TaskIntent, ToolCallRecord


def _plan():
    return Plan(
        intent=TaskIntent.CALCULATION,
        steps=[PlanStep(tool_name="calculator", tool_input={"expression": "1+1"}, reason="test")],
        reasoning="test plan",
    )


def test_evaluator_deterministic_failure_when_all_tools_fail():
    evaluator = Evaluator(llm=None)
    calls = [ToolCallRecord(tool_name="calculator", input={}, success=False, error="boom")]
    result = evaluator.evaluate("1+1", _plan(), calls)
    assert result.status == "FAILURE"
    assert result.score == 0.0


def test_evaluator_deterministic_failure_when_plan_has_steps_but_no_calls():
    evaluator = Evaluator(llm=None)
    result = evaluator.evaluate("1+1", _plan(), tool_calls=[])
    assert result.status == "FAILURE"


def test_evaluator_template_success_when_llm_unavailable():
    evaluator = Evaluator(llm=None)
    calls = [ToolCallRecord(tool_name="calculator", input={}, success=True, output=2.0)]
    result = evaluator.evaluate("1+1", _plan(), calls)
    assert result.status == "SUCCESS"
    assert result.score > 0


class _FakeFailingLLM:
    """An LLM that would (wrongly) claim success - must never override a hard failure."""

    def complete(self, *args, **kwargs):
        from app.llm.base import LLMResponse

        return LLMResponse(content="STATUS: SUCCESS\nSCORE: 1.0\nREASON: looks great\nLESSON: none", model="fake")


def test_llm_cannot_override_deterministic_failure():
    evaluator = Evaluator(llm=_FakeFailingLLM())
    calls = [ToolCallRecord(tool_name="calculator", input={}, success=False, error="boom")]
    result = evaluator.evaluate("1+1", _plan(), calls)
    # Even though the fake LLM says SUCCESS, the deterministic layer runs first
    # and short-circuits - this is the "hard failures must be hard" guarantee.
    assert result.status == "FAILURE"
