from __future__ import annotations

from app.knowledge.retriever import KnowledgeRetriever
from app.tools.base import Tool, ToolResult


class KnowledgeSearchTool(Tool):
    name = "knowledge_search"
    description = "Search the user's uploaded knowledge base for relevant context."
    risk_level = "LOW"
    timeout_seconds = 5
    input_schema = {"query": "string", "top_k": "integer, default 4"}
    output_schema = {"output": "object - {chunks: [{text, similarity}]}"}

    def __init__(self, retriever: KnowledgeRetriever | None = None):
        self._retriever = retriever or KnowledgeRetriever()

    def _run(self, query: str = "", top_k: int = 4, **kwargs) -> ToolResult:
        if not query.strip():
            return ToolResult(success=False, error="No query provided.")
        chunks = self._retriever.search(query, top_k=top_k)
        if not chunks:
            return ToolResult(success=True, output={"chunks": [], "message": "Knowledge base is empty."})
        return ToolResult(
            success=True,
            output={"chunks": [{"text": c.text, "similarity": c.similarity} for c in chunks]},
        )