"""
Application configuration.

All settings are read from environment variables (or a `.env` file in the
project root). Nothing in this file should ever contain a real secret -
see `.env.example` for the list of variables a deployer needs to set.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=("settings_",))

    # --- LLM provider -----------------------------------------------------
    llm_provider: str = "anthropic"  # "anthropic" | "openai" | "gemini"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    model_name: str = "claude-sonnet-4-5-20250929"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # --- Database -----------------------------------------------------------
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'db' / 'agent.db'}"

    # --- Vector memory ------------------------------------------------------
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma")
    embedding_provider: str = "hashing"  # "hashing" | "openai"

    # --- Agent limits (see app/agent/state.py) -------------------------------
    max_iterations: int = 6
    max_tool_calls: int = 8
    max_retries: int = 2
    tool_timeout_seconds: int = 10

    # --- Knowledge / uploads --------------------------------------------------
    upload_dir: str = str(BASE_DIR / "data" / "uploads")
    max_upload_mb: int = 20
    allowed_upload_extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".csv")
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Misc -----------------------------------------------------------------
    app_name: str = "Self-Learning AI Agent"
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173")
    rate_limit_per_minute: int = 30


settings = Settings()

# Ensure runtime directories exist.
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(BASE_DIR / "data" / "db").mkdir(parents=True, exist_ok=True)
