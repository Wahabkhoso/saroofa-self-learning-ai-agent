"""
Semantic memory: find past experiences that are conceptually similar to
the current task, not just keyword matches.

Storage: ChromaDB, persisted to disk under `CHROMA_PERSIST_DIR`.
Embeddings: pluggable, see app/knowledge/embeddings.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from app.core.config import settings
from app.core.logging_config import get_logger
from app.knowledge.embeddings import ChromaEmbeddingAdapter, EmbeddingProvider, get_embedding_provider

logger = get_logger(__name__)


@dataclass
class RetrievedExperience:
    experience_id: str
    task_summary: str
    lesson: str
    solution: str
    outcome: str
    confidence: float
    similarity: float
    # Populated by LongTermMemory's ranking step (not by the raw vector
    # search) - see app/memory/long_term.py::_rank.
    usefulness_score: float = 0.5
    rank_score: float = 0.0


class VectorMemory:
    """Wraps a ChromaDB collection storing embedded experience summaries."""

    COLLECTION_NAME = "experiences"

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

    def add_experience(
        self,
        *,
        experience_id: str,
        task_summary: str,
        lesson: str,
        solution: str,
        outcome: str,
        confidence: float,
    ) -> None:
        self._collection.upsert(
            ids=[experience_id],
            documents=[task_summary],
            metadatas=[
                {
                    "lesson": lesson,
                    "solution": solution[:2000],
                    "outcome": outcome,
                    "confidence": confidence,
                }
            ],
        )
        logger.info("vector_memory: stored experience %s", experience_id)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedExperience]:
        if self._collection.count() == 0:
            return []
        top_k = min(top_k, self._collection.count())
        results = self._collection.query(query_texts=[query], n_results=top_k)

        retrieved: list[RetrievedExperience] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for exp_id, doc, meta, distance in zip(ids, docs, metas, distances):
            similarity = max(0.0, 1.0 - distance)  # cosine distance -> similarity
            retrieved.append(
                RetrievedExperience(
                    experience_id=exp_id,
                    task_summary=doc,
                    lesson=meta.get("lesson", ""),
                    solution=meta.get("solution", ""),
                    outcome=meta.get("outcome", ""),
                    confidence=float(meta.get("confidence", 0.5)),
                    similarity=round(similarity, 4),
                )
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()