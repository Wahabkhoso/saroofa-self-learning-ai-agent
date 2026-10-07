"""
Embedding providers for semantic search.

Design note (read this before assuming this is "fake"):
Real embedding models (OpenAI text-embedding-3, sentence-transformers, etc.)
require either an API key or a multi-hundred-MB model download at first
run. To keep this project runnable offline, with zero setup, and fully
testable in CI, the default provider is a deterministic *hashing
vectorizer* (a bag-of-words / feature-hashing scheme, TF weighted,
L2-normalized). It is a real, well-understood technique - not a mock -
but it is weaker than a trained embedding model: it captures lexical
overlap, not deep semantic meaning. For production use, swap in
`OpenAIEmbeddingProvider` (or add a sentence-transformers provider) by
changing `EMBEDDING_PROVIDER` in `.env`. Both implement the same
interface, so nothing else in the codebase needs to change.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.core.config import settings

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


class EmbeddingProvider(ABC):
    dimensions: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free, offline embedding via feature hashing."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text)]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % self.dimensions
            sign = 1.0 if int(digest[:8], 16) % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Real embedding model via the OpenAI API. Requires OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        from openai import OpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to use OpenAIEmbeddingProvider")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding


def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider()
    return HashingEmbeddingProvider()


class ChromaEmbeddingAdapter:
    """Adapts our EmbeddingProvider interface to ChromaDB's EmbeddingFunction protocol.

    Shared by both vector memory collections (experiences, knowledge) so
    there is exactly one adapter implementation.

    chromadb >=1.x requires embedding functions to implement `name()`,
    `get_config()`, and `build_from_config()` (not just `__call__`) - see
    chromadb.api.types.EmbeddingFunction. We keep `get_config`/
    `build_from_config` minimal since this adapter is always constructed
    explicitly by our own code (VectorMemory / KnowledgeRetriever), not
    reconstructed by chromadb from persisted config.
    """

    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._provider.embed_batch(list(input))

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        # chromadb calls this for query-time embedding when it isn't the
        # default embedding function; we don't distinguish query vs.
        # document embeddings, so just delegate to the same method.
        return self.__call__(input)

    def name(self) -> str:
        return f"custom-{type(self._provider).__name__}"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "ChromaEmbeddingAdapter":
        # Only reached if chromadb reconstructs this from persisted config
        # without an explicit embedding_function being passed in - our code
        # always passes one explicitly, so this is a safe fallback.
        return ChromaEmbeddingAdapter(get_embedding_provider())