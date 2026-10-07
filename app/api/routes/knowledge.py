from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_knowledge_retriever
from app.core.config import settings
from app.database.repositories import DocumentRepository
from app.knowledge.loader import load_text
from app.knowledge.retriever import KnowledgeRetriever
from app.schemas.knowledge import DocumentExtractResponse, DocumentOut, DocumentUploadResponse

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
MAX_EXTRACT_CHARS = 8000


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name  # strips any directory traversal component
    name = _SAFE_NAME_RE.sub("_", name)
    return name or "upload"


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    retriever: KnowledgeRetriever = Depends(get_knowledge_retriever),
) -> DocumentUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_upload_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.allowed_upload_extensions}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413, detail=f"File too large ({size_mb:.1f}MB > {settings.max_upload_mb}MB limit)."
        )

    safe_name = _sanitize_filename(file.filename or f"{uuid.uuid4().hex}{suffix}")
    dest_path = Path(settings.upload_dir) / safe_name
    # Avoid overwriting an existing file with the same name.
    if dest_path.exists():
        safe_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        dest_path = Path(settings.upload_dir) / safe_name
    dest_path.write_bytes(contents)

    doc_repo = DocumentRepository(db)
    document = doc_repo.create(
        filename=safe_name, file_type=suffix.lstrip("."), size_bytes=len(contents), chunk_count=0
    )

    try:
        chunk_count = retriever.ingest(document.id, dest_path)
    except Exception as exc:  # noqa: BLE001
        doc_repo.delete(document.id)
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to process document: {exc}") from exc

    document.chunk_count = chunk_count
    db.commit()
    db.refresh(document)

    return DocumentUploadResponse(
        document=DocumentOut.model_validate(document),
        message=f"Ingested {chunk_count} chunks from '{safe_name}'.",
    )


@router.post("/extract", response_model=DocumentExtractResponse)
async def extract_document_text(file: UploadFile) -> DocumentExtractResponse:
    """Extract raw text from an uploaded file WITHOUT saving it to disk or
    adding it to the persistent knowledge base. Used when a document is
    attached directly to a chat message (one-off context for that message
    only) - see the frontend's 'Document' attach option. Permanently
    ingesting a file for RAG search still goes through /upload."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_upload_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.allowed_upload_extensions}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413, detail=f"File too large ({size_mb:.1f}MB > {settings.max_upload_mb}MB limit)."
        )

    if suffix in (".pdf", ".docx"):
        # load_text() only accepts a path on disk (same extractor the RAG
        # pipeline and file_inspector tool use) - write to a throwaway temp
        # file just long enough to extract text, then remove it. This
        # endpoint never persists the upload; /upload does that instead.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        try:
            text = load_text(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        text = contents.decode("utf-8", errors="replace")

    truncated = len(text) > MAX_EXTRACT_CHARS
    return DocumentExtractResponse(
        filename=_sanitize_filename(file.filename or f"document{suffix}"),
        text=text[:MAX_EXTRACT_CHARS],
        truncated=truncated,
        length=len(text),
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentOut]:
    return [DocumentOut.model_validate(d) for d in DocumentRepository(db).list_all()]


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    retriever: KnowledgeRetriever = Depends(get_knowledge_retriever),
) -> dict:
    repo = DocumentRepository(db)
    doc = repo.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(settings.upload_dir) / doc.filename
    file_path.unlink(missing_ok=True)
    retriever.delete_document(document_id)
    repo.delete(document_id)
    return {"deleted": True, "id": document_id}