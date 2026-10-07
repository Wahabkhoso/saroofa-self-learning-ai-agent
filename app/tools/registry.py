from __future__ import annotations

from app.tools.base import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.file_tool import FileTool
from app.tools.knowledge_search import KnowledgeSearchTool
from app.tools.python_tool import PythonTool


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(PythonTool())
    registry.register(FileTool())
    registry.register(KnowledgeSearchTool())
    return registry
