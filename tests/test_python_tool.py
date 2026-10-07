from app.tools.python_tool import PythonTool, _check_ast_safety, PythonSafetyError
import pytest


def test_ast_safety_allows_whitelisted_import():
    _check_ast_safety("import math\nprint(math.sqrt(4))")


def test_ast_safety_blocks_os_import():
    with pytest.raises(PythonSafetyError):
        _check_ast_safety("import os\nos.system('ls')")


def test_ast_safety_blocks_eval():
    with pytest.raises(PythonSafetyError):
        _check_ast_safety("eval('1+1')")


def test_ast_safety_blocks_dunder_access():
    with pytest.raises(PythonSafetyError):
        _check_ast_safety("().__class__.__bases__")


def test_ast_safety_blocks_write_mode_open():
    with pytest.raises(PythonSafetyError):
        _check_ast_safety("open('x.txt', 'w')")


def test_python_tool_runs_simple_code():
    tool = PythonTool()
    result = tool.run(code="print(2 + 2)")
    assert result.success is True
    assert result.output == "4"


def test_python_tool_blocks_dangerous_code():
    tool = PythonTool()
    result = tool.run(code="import subprocess; subprocess.run(['ls'])")
    assert result.success is False
    assert "sandbox policy" in result.error.lower()


def test_python_tool_timeout(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "tool_timeout_seconds", 1)
    tool = PythonTool()
    result = tool.run(code="while True:\n    pass")
    assert result.success is False
    assert "timed out" in result.error.lower()
