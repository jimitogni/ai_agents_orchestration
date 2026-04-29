"""Request and response models for API routes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    question: str = Field(min_length=1, examples=["What is overfitting?"])


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    ingested_files: list[str]
    chunks_indexed: int
    collection: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    services: dict[str, str]
    environment: str

