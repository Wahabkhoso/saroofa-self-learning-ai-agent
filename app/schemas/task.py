from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


_ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
_MAX_IMAGE_BASE64_CHARS = 15_000_000  # ~10-11MB image, generous for a screenshot/photo
_MAX_DOCUMENT_TEXT_CHARS = 20_000


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=8000)
    session_id: str = Field(default="default")
    image_base64: str | None = Field(
        default=None, max_length=_MAX_IMAGE_BASE64_CHARS,
        description="Base64-encoded image, no data: prefix",
    )
    image_mime_type: str | None = Field(default=None, description="e.g. 'image/png', 'image/jpeg'")
    document_text: str | None = Field(
        default=None, max_length=_MAX_DOCUMENT_TEXT_CHARS,
        description="Extracted text of a document attached to this message (see /api/knowledge/extract)",
    )
    document_filename: str | None = Field(default=None, description="Original filename of the attached document")

    @model_validator(mode="after")
    def _require_text_or_attachment(self) -> "ChatRequest":
        if not self.message.strip() and not self.image_base64 and not self.document_text:
            raise ValueError("Provide a message, an image, a document, or some combination.")
        if self.image_base64 and self.image_mime_type not in _ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError(f"image_mime_type must be one of {sorted(_ALLOWED_IMAGE_MIME_TYPES)}")
        return self


class ToolCallOut(BaseModel):
    tool_name: str
    success: bool
    output: object | None = None
    error: str | None = None
    duration_seconds: float


class EvaluationOut(BaseModel):
    status: str
    score: float
    confidence: float
    reason: str
    lesson: str


class TraceStepOut(BaseModel):
    name: str
    status: str
    duration_seconds: float | None = None
    metadata: dict = {}
    error: str | None = None


class ChatResponse(BaseModel):
    task_id: str
    answer: str
    intent: str
    activity_log: list[str]
    trace: list[TraceStepOut]
    tool_calls: list[ToolCallOut]
    evaluation: EvaluationOut
    retries: int
    retrieved_experiences: list[str]
    experience_id: str | None = None
    agent_confidence: float
    failure_category: str | None = None


class TaskOut(BaseModel):
    id: str
    session_id: str
    user_input: str
    status: str
    result: str | None
    score: float | None
    confidence: float | None
    agent_confidence: float | None = None
    error: str | None
    retries: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionOut(BaseModel):
    session_id: str
    title: str
    task_count: int
    last_activity: datetime