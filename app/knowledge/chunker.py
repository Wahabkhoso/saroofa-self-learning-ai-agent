"""
Splits long text into overlapping chunks suitable for embedding.

Uses a simple character-based sliding window that tries to break on
paragraph/sentence boundaries where possible, rather than mid-word.
"""
from __future__ import annotations

from app.core.config import settings


def chunk_text(
    text: str, chunk_size: int | None = None, overlap: int | None = None
) -> list[str]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]

        if end < len(text):
            # Prefer to break at the last paragraph/sentence/space boundary.
            for boundary in ("\n\n", ". ", "\n", " "):
                idx = window.rfind(boundary)
                if idx > chunk_size * 0.5:
                    end = start + idx + len(boundary)
                    window = text[start:end]
                    break

        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return [c for c in chunks if c]
