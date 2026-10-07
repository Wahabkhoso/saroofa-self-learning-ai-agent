from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.agent.agent import Agent
from app.agent.state import AgentLimits
from app.core.config import settings
from app.database.database import get_db
from app.knowledge.retriever import KnowledgeRetriever
from app.llm.base import LLMProvider, LLMProviderError
from app.llm.factory import get_llm_provider
from app.memory.long_term import LongTermMemory
from app.memory.vector_memory import VectorMemory
from app.tools.registry import build_default_registry

# Process-wide singletons: safe because they're either stateless helpers
# or wrap a persistent store (Chroma/SQLite) that already handles
# concurrent access.
_tool_registry = build_default_registry()
_vector_memory = VectorMemory()
_knowledge_retriever = KnowledgeRetriever()


def get_llm() -> LLMProvider | None:
    try:
        return get_llm_provider()
    except LLMProviderError:
        return None


def get_tool_registry():
    return _tool_registry


def get_knowledge_retriever() -> KnowledgeRetriever:
    return _knowledge_retriever


def get_long_term_memory(db: Session = Depends(get_db)) -> LongTermMemory:
    return LongTermMemory(db, vector_memory=_vector_memory)


def get_agent(
    db: Session = Depends(get_db),
    long_term_memory: LongTermMemory = Depends(get_long_term_memory),
) -> Agent:
    llm = get_llm()
    return Agent(
        long_term_memory=long_term_memory,
        tool_registry=_tool_registry,
        llm=llm,
        limits=AgentLimits(),
    )


# --- Simple in-memory rate limiter (per client IP, sliding window) -----------

_request_log: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    client_id = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = 60.0
    log = _request_log[client_id]

    while log and now - log[0] > window:
        log.popleft()

    if len(log) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down.",
        )
    log.append(now)
