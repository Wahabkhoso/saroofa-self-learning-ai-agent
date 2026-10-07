from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_tool_registry
from app.tools.base import ToolRegistry

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("")
def list_tools(registry: ToolRegistry = Depends(get_tool_registry)) -> list[dict]:
    """Tool Registry metadata (spec section 10) - name, description,
    risk level, timeout, schemas. Used by the Planner internally and
    exposed here for transparency/debugging."""
    return registry.list_metadata()