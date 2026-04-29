"""Small HTTP client for the Ollama API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama text generation."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def is_healthy(self) -> bool:
        """Return True when Ollama is reachable and the configured model is installed."""
        try:
            with httpx.Client(transport=self._transport) as client:
                response = client.get(
                    f"{self.base_url}/api/tags",
                    timeout=5.0,
                )
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = {
                str(model.get("name") or model.get("model"))
                for model in models
                if isinstance(model, dict)
            }
            return self.model in model_names
        except httpx.HTTPError as exc:
            logger.debug("Ollama health check failed: %s", exc)
            return False

    def generate(self, prompt: str) -> str:
        """Generate an answer from the configured Ollama model."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }
        with httpx.Client(transport=self._transport) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()

        data = response.json()
        answer = str(data.get("response", "")).strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        return answer
