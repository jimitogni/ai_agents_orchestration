"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the API and backing services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Enterprise RAG Agent"
    app_environment: str = "local"
    log_level: str = "INFO"

    docs_path: Path = Path("data/docs")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 120.0

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "internal_docs"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 4
    chunk_size: int = 900
    chunk_overlap: int = 120


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()
