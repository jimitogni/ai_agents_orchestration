"""Tests for the health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


class HealthyService:
    """Fake healthy external service."""

    def is_healthy(self) -> bool:
        return True


def test_health_endpoint_reports_all_services_ok() -> None:
    app.dependency_overrides[routes.get_ollama_client] = lambda: HealthyService()
    app.dependency_overrides[routes.get_chroma_service] = lambda: HealthyService()

    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["services"]["api"] == "ok"
    assert body["services"]["ollama"] == "ok"
    assert body["services"]["chromadb"] == "ok"

