"""
Application entrypoint.

Run with:  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/ for the web UI, or /docs for the API reference.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, feedback, knowledge, memory, stats, tasks
from app.core.config import BASE_DIR, settings
from app.core.logging_config import configure_logging, get_logger
from app.database.database import init_db

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (LLM provider: %s)", settings.app_name, settings.llm_provider)
    init_db()
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description=(
        "A self-learning AI agent. It does not retrain the underlying LLM - "
        "it improves via persistent memory, semantic retrieval, feedback and "
        "evaluated experience. See README for details."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(tasks.router)
app.include_router(memory.router)
app.include_router(feedback.router)
app.include_router(knowledge.router)
app.include_router(stats.router)

_frontend_dir = BASE_DIR / "frontend"
if _frontend_dir.exists():
    # Mounted last and at "/" so it never shadows the /api/* routes above,
    # while still serving the UI at the site root for a one-URL demo.
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
