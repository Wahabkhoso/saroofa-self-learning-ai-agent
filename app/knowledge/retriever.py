"""
Knowledge base retriever (RAG).

Pipeline: load document -> extract text -> chunk -> embed -> store in a
dedicated Chroma collection ("knowledge", separate from "experiences") ->
retrieve top-k relevant chunks for a query at answer time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.core.config import settings
from app.core.logging_config import get_logger
from app.knowledge.chunker import chunk_text
from app.knowledge.embeddings import ChromaEmbeddingAdapter, EmbeddingProvider, get_embedding_provider
from app.knowledge.loader import load_text

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    document_id: str
    chunk_index: int
    text: str
    similarity: float


class KnowledgeRetriever:
    COLLECTION_NAME = "knowledge"

    def __init__(self, embedding_provider: EmbeddingProvider | None = None):
        provider = embedding_provider or get_embedding_provider()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=ChromaEmbeddingAdapter(provider),
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, document_id: str, path: Path) -> int:
        """Loads, chunks, embeds and stores a document. Returns chunk count."""
        text = load_text(path)
        chunks = chunk_text(text)
        if not chunks:
            return 0

        ids = [f"{document_id}::{i}" for i in range(len(chunks))]
        metadatas = [{"document_id": document_id, "chunk_index": i} for i in range(len(chunks))]
        self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        logger.info("knowledge_retriever: ingested %s chunks for doc %s", len(chunks), document_id)
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []
        top_k = min(top_k, self._collection.count())
        results = self._collection.query(query_texts=[query], n_results=top_k)

        chunks: list[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc, meta, distance in zip(docs, metas, distances):
            chunks.append(
                RetrievedChunk(
                    document_id=meta.get("document_id", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    text=doc,
                    similarity=round(max(0.0, 1.0 - distance), 4),
                )
            )
        return chunks
