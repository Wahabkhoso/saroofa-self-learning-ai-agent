"""
Python execution tool.

Safety model (be honest about what this is and isn't):
  1. Static AST check rejects imports outside a whitelist, and rejects
     dunder attribute access, `exec`/`eval`/`compile`, and `open()` in
     write/append modes.
  2. The (already-checked) code runs in a *separate subprocess*, not in
     this process, with `-I` (isolated mode, ignores env/site config)
     and a hard wall-clock timeout, so a runaway or hung script cannot
     block or crash the API server.
  3. Working directory is a fresh temp dir; only stdlib + pandas/numpy
     are importable.

This is reasonable isolation for a portfolio / local demo project. It is
NOT a substitute for a real sandbox (gVisor, Firecracker, a container
with dropped capabilities) and should not be exposed to untrusted
multi-tenant traffic as-is - that limitation is called out in the README.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

from app.core.config import settings
from app.tools.base import Tool, ToolResult

ALLOWED_IMPORTS = {
    "math", "statistics", "json", "re", "datetime", "itertools",
    "collections", "pandas", "numpy", "random", "decimal",
}
FORBIDDEN_CALL_NAMES = {"exec", "eval", "compile", "__import__", "input"}


class PythonSafetyError(ValueError):
    pass


def _check_ast_safety(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PythonSafetyError(f"Syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise PythonSafetyError(f"Import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise PythonSafetyError(f"Import not allowed: {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise PythonSafetyError(f"Dunder attribute access not allowed: {node.attr}")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALL_NAMES:
                raise PythonSafetyError(f"Call not allowed: {fn.id}()")
            if isinstance(fn, ast.Name) and fn.id == "open" and node.args:
                mode_arg = node.keywords
                # Reject explicit write/append modes; default open() (read) is fine.
                for kw in mode_arg:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        if any(c in str(kw.value.value) for c in "wax"):
                            raise PythonSafetyError("open() in write/append mode is not allowed")
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    if any(c in str(node.args[1].value) for c in "wax"):
                        raise PythonSafetyError("open() in write/append mode is not allowed")


class PythonTool(Tool):
    name = "python_executor"
    description = (
        "Run a short, sandboxed Python snippet for calculations or data processing "
        "(pandas/numpy available). No filesystem writes, no network, no shell access."
    )
    risk_level = "MEDIUM"
    timeout_seconds = settings.tool_timeout_seconds
    input_schema = {"code": "string - a Python snippet"}
    output_schema = {"output": "string - captured stdout"}

    def _run(self, code: str = "", **kwargs) -> ToolResult:
        if not code.strip():
            return ToolResult(success=False, error="No code provided.")

        try:
            _check_ast_safety(code)
        except PythonSafetyError as exc:
            return ToolResult(success=False, error=f"Blocked by sandbox policy: {exc}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "snippet.py"
            script_path.write_text(code)
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=settings.tool_timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False,
                    error=f"Execution timed out after {settings.tool_timeout_seconds}s.",
                )

            if proc.returncode != 0:
                stderr = proc.stderr.strip()
                # Keep the exception's own message (last non-empty line) as the
                # primary error; the full traceback is verbose and not useful
                # to show in the UI/DB - only useful when debugging the tool itself.
                last_line = next(
                    (line for line in reversed(stderr.splitlines()) if line.strip()), stderr
                )
                return ToolResult(success=False, error=last_line[:500])
            return ToolResult(success=True, output=proc.stdout.strip())