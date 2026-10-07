"""
Calculator tool.

Deliberately does NOT use eval(). Expressions are parsed into an AST and
walked by hand, allowing only numeric literals, arithmetic operators and
a small whitelist of math functions. Anything else (attribute access,
names, calls to unknown functions, comprehensions, etc.) is rejected
before any evaluation happens.
"""
from __future__ import annotations

import ast
import math
import operator

from app.tools.base import Tool, ToolResult

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
}
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculatorError(ValueError):
    pass


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise CalculatorError("Only whitelisted math functions are allowed")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise CalculatorError(f"Unknown identifier: {node.id}")
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression: {exc}") from exc
    return _eval_node(tree)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a mathematical expression safely (e.g. '(12 + 8) * 3 / sqrt(4)')."
    risk_level = "LOW"
    timeout_seconds = 1
    input_schema = {"expression": "string - a math expression"}
    output_schema = {"output": "number"}

    def _run(self, expression: str = "", **kwargs) -> ToolResult:
        if not expression.strip():
            return ToolResult(success=False, error="No expression provided.")
        try:
            value = safe_eval(expression)
        except CalculatorError as exc:
            return ToolResult(success=False, error=str(exc))
        except ZeroDivisionError:
            return ToolResult(success=False, error="Division by zero.")
        return ToolResult(success=True, output=value)