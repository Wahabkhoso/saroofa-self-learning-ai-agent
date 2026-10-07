from app.agent.planner import Planner, TaskAnalyzer, _extract_expression
from app.agent.state import TaskIntent


def test_analyzer_classifies_calculation():
    assert TaskAnalyzer().analyze("Calculate (12 + 8) * 3 / sqrt(4)") == TaskIntent.CALCULATION


def test_analyzer_classifies_csv():
    assert (
        TaskAnalyzer().analyze("I need to analyze students.csv and calculate the average salary")
        == TaskIntent.CSV_OR_DATA_ANALYSIS
    )


def test_analyzer_classifies_file_read():
    assert TaskAnalyzer().analyze("Summarize notes.txt for me") == TaskIntent.FILE_READ


def test_analyzer_classifies_general():
    assert TaskAnalyzer().analyze("What's a good way to learn Python?") == TaskIntent.GENERAL


def test_extract_expression_strips_leading_command_word():
    # Regression test: this used to leak the word "Calculate" into the
    # expression string, which made the calculator reject it as an
    # unknown function call.
    expr = _extract_expression("Calculate (12 + 8) * 3 / sqrt(4)")
    assert "calculate" not in expr.lower()
    assert "sqrt" in expr


def test_csv_planner_extracts_filename_without_swallowing_sentence():
    # Regression test: the old regex greedily matched the whole leading
    # sentence, not just the filename.
    plan = Planner().build_plan(
        "I need to analyze students.csv and calculate the average salary",
        TaskIntent.CSV_OR_DATA_ANALYSIS,
    )
    assert len(plan.steps) == 1
    code = plan.steps[0].tool_input["code"]
    assert "students.csv" in code
    assert "I need to analyze" not in code


def test_csv_planner_detects_column_hint():
    plan = Planner().build_plan(
        "Analyze payroll.csv and calculate the average bonus", TaskIntent.CSV_OR_DATA_ANALYSIS
    )
    assert "bonus" in plan.steps[0].tool_input["code"].lower()


def test_calculation_planner_produces_clean_expression():
    plan = Planner().build_plan("What is 2 + 2?", TaskIntent.CALCULATION)
    assert plan.steps[0].tool_input["expression"].strip() != ""
    assert "what" not in plan.steps[0].tool_input["expression"].lower()
