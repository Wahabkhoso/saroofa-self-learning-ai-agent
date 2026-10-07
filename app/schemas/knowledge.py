from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    document: DocumentOut
    message: str


class DocumentExtractResponse(BaseModel):
    """Response for /api/knowledge/extract - a one-off text extraction that
    is NOT saved to disk or added to the persistent knowledge base. Used
    when the user attaches a document directly to a chat message (as
    opposed to /upload, which permanently ingests a file for RAG search)."""

    filename: str
    text: str
    truncated: bool
    length: int