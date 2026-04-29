"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Local enterprise-style AI agent with FastAPI, LangGraph, Ollama, and ChromaDB."
        ),
    )
    application.include_router(router)
    return application


app = create_app()
