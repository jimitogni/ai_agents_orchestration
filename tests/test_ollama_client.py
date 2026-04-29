"""Tests for the Ollama client."""

from __future__ import annotations

import httpx

from app.services.ollama_client import OllamaClient


def test_ollama_health_requires_configured_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            status_code=200,
            json={"models": [{"name": "llama3.2:3b"}]},
        )

    client = OllamaClient(
        base_url="http://ollama.test",
        model="llama3.2:3b",
        transport=httpx.MockTransport(handler),
    )

    assert client.is_healthy() is True


def test_ollama_health_fails_when_model_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(status_code=200, json={"models": []})

    client = OllamaClient(
        base_url="http://ollama.test",
        model="llama3.2:3b",
        transport=httpx.MockTransport(handler),
    )

    assert client.is_healthy() is False
