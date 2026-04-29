"""HTTP routes for chat, ingestion, and service health."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.agents.graph import RagAgent
from app.core.config import Settings, get_settings
from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from app.services.chroma_client import ChromaService
from app.services.ingestion_service import IngestionService
from app.services.ollama_client import OllamaClient
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache
def get_chroma_service() -> ChromaService:
    """Create a cached ChromaDB service."""
    settings = get_settings()
    return ChromaService(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache
def get_ollama_client() -> OllamaClient:
    """Create a cached Ollama client."""
    settings = get_settings()
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """Create a cached retrieval service."""
    settings = get_settings()
    return RetrievalService(
        chroma_service=get_chroma_service(),
        embedding_model_name=settings.embedding_model_name,
        top_k=settings.retrieval_top_k,
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    """Create a cached document ingestion service."""
    settings = get_settings()
    return IngestionService(
        chroma_service=get_chroma_service(),
        docs_path=settings.docs_path,
        embedding_model_name=settings.embedding_model_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_agent() -> RagAgent:
    """Create a cached LangGraph RAG agent."""
    return RagAgent(
        retrieval_service=get_retrieval_service(),
        ollama_client=get_ollama_client(),
    )


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
    ollama_client: Annotated[OllamaClient, Depends(get_ollama_client)],
    chroma_service: Annotated[ChromaService, Depends(get_chroma_service)],
) -> HealthResponse:
    """Return health information for the API and external services."""
    services = {
        "api": "ok",
        "ollama": "ok" if ollama_client.is_healthy() else "unavailable",
        "chromadb": "ok" if chroma_service.is_healthy() else "unavailable",
    }
    overall_status = "ok" if all(value == "ok" for value in services.values()) else "degraded"
    return HealthResponse(
        status=overall_status,
        services=services,
        environment=settings.app_environment,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    ingestion_service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestResponse:
    """Ingest local markdown and text documents into ChromaDB."""
    try:
        result = ingestion_service.ingest_documents()
    except FileNotFoundError as exc:
        logger.warning("Document ingestion failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document ingestion failed. Check API logs and dependent services.",
        ) from exc

    return IngestResponse(
        ingested_files=result.ingested_files,
        chunks_indexed=result.chunks_indexed,
        collection=result.collection,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    agent: Annotated[RagAgent, Depends(get_agent)],
) -> ChatResponse:
    """Answer a question using the LangGraph RAG workflow."""
    try:
        result = agent.answer(request.question)
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The agent could not answer right now. Check Ollama and ChromaDB availability.",
        ) from exc

    return ChatResponse(answer=result.answer, sources=result.sources)
