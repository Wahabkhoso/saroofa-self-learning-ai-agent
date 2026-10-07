"""
Shared test fixtures.

Every test gets its own temp SQLite DB and temp Chroma directory, so
tests never interfere with each other or with real local data.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def temp_env(monkeypatch, tmp_path):
    """Point the app at fresh, isolated storage for the duration of one test.

    Uses monkeypatch.setattr (not raw assignment) everywhere so pytest
    automatically reverts every change at teardown - this is what keeps
    tests from leaking state into each other via the shared `settings`
    singleton.
    """
    from app.core.config import settings

    db_dir = tmp_path / "db"
    chroma_dir = tmp_path / "chroma"
    upload_dir = tmp_path / "uploads"
    db_dir.mkdir()
    chroma_dir.mkdir()
    upload_dir.mkdir()

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_dir / 'test.db'}")
    monkeypatch.setattr(settings, "chroma_persist_dir", str(chroma_dir))
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "embedding_provider", "hashing")

    yield {"db_dir": db_dir, "chroma_dir": chroma_dir, "upload_dir": upload_dir}


@pytest.fixture()
def db_session(temp_env):
    from app.core.config import settings
    from app.database.database import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def api_client(monkeypatch, temp_env):
    """A TestClient wired to fully isolated, temporary storage.

    This has to patch more than `db_session` does: the FastAPI app builds
    a few process-wide singletons (DB engine/session factory, tool
    registry, vector memory, knowledge retriever) once at import time in
    app/api/dependencies.py. For a test to be truly isolated, all of
    those need to point at this test's temp directory too - not just the
    `settings` object they originally read from.
    """
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.config import settings
    from app.database.database import Base
    import app.database.database as database_module
    import app.api.dependencies as deps_module
    from app.knowledge.retriever import KnowledgeRetriever
    from app.memory.vector_memory import VectorMemory
    from app.tools.registry import build_default_registry

    test_engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)

    monkeypatch.setattr(database_module, "engine", test_engine)
    monkeypatch.setattr(database_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(deps_module, "_vector_memory", VectorMemory())
    monkeypatch.setattr(deps_module, "_knowledge_retriever", KnowledgeRetriever())
    monkeypatch.setattr(deps_module, "_tool_registry", build_default_registry())

    from app.main import app

    with TestClient(app) as client:
        yield client

    test_engine.dispose()
