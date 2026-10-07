"""
File tool: read-only inspection of files the user has uploaded.

Safety:
  - Path is resolved and must stay inside the configured upload directory
    (blocks `../../etc/passwd`-style traversal).
  - Only a whitelist of extensions is readable.
  - Output is truncated so a huge file can't blow up the LLM context.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from app.core.config import settings
from app.tools.base import Tool, ToolResult

MAX_PREVIEW_CHARS = 4000


class FileTool(Tool):
    name = "file_inspector"
    description = (
        "Inspect an uploaded file: preview text, summarize CSV columns/rows, "
        "or pretty-print JSON structure. Read-only."
    )
    risk_level = "LOW"
    timeout_seconds = 5
    input_schema = {"filename": "string - a file already in the uploads directory"}
    output_schema = {"output": "object - preview/summary, shape depends on file type"}

    def _resolve_safe_path(self, filename: str) -> Path:
        base = Path(settings.upload_dir).resolve()
        candidate = (base / filename).resolve()
        if base not in candidate.parents and candidate != base:
            raise ValueError("Path escapes the uploads directory.")
        if not candidate.is_relative_to(base):
            raise ValueError("Path escapes the uploads directory.")
        return candidate

    def _run(self, filename: str = "", **kwargs) -> ToolResult:
        if not filename:
            return ToolResult(success=False, error="No filename provided.")
        try:
            path = self._resolve_safe_path(filename)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {filename}")
        if path.suffix.lower() not in settings.allowed_upload_extensions:
            return ToolResult(success=False, error=f"Unsupported file type: {path.suffix}")

        try:
            if path.suffix.lower() == ".csv":
                return self._inspect_csv(path)
            if path.suffix.lower() == ".json":
                return self._inspect_json(path)
            if path.suffix.lower() in (".pdf", ".docx"):
                # These are binary/structured formats, not plain text -
                # route through the same extractor the RAG pipeline uses
                # (app/knowledge/loader.py) rather than reading raw bytes.
                return self._inspect_extracted(path)
            return self._inspect_text(path)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Failed to read file: {exc}")

    def _inspect_extracted(self, path: Path) -> ToolResult:
        from app.knowledge.loader import load_text

        text = load_text(path)
        preview = text[:MAX_PREVIEW_CHARS]
        return ToolResult(
            success=True,
            output={"preview": preview, "truncated": len(text) > MAX_PREVIEW_CHARS, "length": len(text)},
        )

    def _inspect_text(self, path: Path) -> ToolResult:
        text = path.read_text(errors="replace")
        preview = text[:MAX_PREVIEW_CHARS]
        return ToolResult(
            success=True,
            output={"preview": preview, "truncated": len(text) > MAX_PREVIEW_CHARS, "length": len(text)},
        )

    def _inspect_csv(self, path: Path) -> ToolResult:
        with path.open(newline="", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)
        header = rows[0] if rows else []
        data_rows = rows[1:]
        return ToolResult(
            success=True,
            output={
                "columns": header,
                "row_count": len(data_rows),
                "preview_rows": data_rows[:5],
            },
        )

    def _inspect_json(self, path: Path) -> ToolResult:
        data = json.loads(path.read_text(errors="replace"))
        if isinstance(data, list):
            summary = {"type": "list", "length": len(data), "sample": data[:3]}
        elif isinstance(data, dict):
            summary = {"type": "object", "keys": list(data.keys())[:20]}
        else:
            summary = {"type": type(data).__name__, "value": data}
        return ToolResult(success=True, output=summary)