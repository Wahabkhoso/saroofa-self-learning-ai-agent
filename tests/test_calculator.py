from app.tools.calculator import CalculatorTool, safe_eval, CalculatorError
import pytest


def test_safe_eval_basic_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14


def test_safe_eval_functions():
    assert safe_eval("sqrt(16)") == 4.0


def test_safe_eval_rejects_names():
    with pytest.raises(CalculatorError):
        safe_eval("os.system('ls')")


def test_safe_eval_rejects_unknown_function():
    with pytest.raises(CalculatorError):
        safe_eval("Calculate(4)")


def test_calculator_tool_success():
    tool = CalculatorTool()
    result = tool.run(expression="(12 + 8) * 3 / sqrt(4)")
    assert result.success is True
    assert result.output == 30.0


def test_calculator_tool_division_by_zero():
    tool = CalculatorTool()
    result = tool.run(expression="1/0")
    assert result.success is False
    assert "zero" in result.error.lower()


def test_calculator_tool_empty_expression():
    tool = CalculatorTool()
    result = tool.run(expression="")
    assert result.success is False
