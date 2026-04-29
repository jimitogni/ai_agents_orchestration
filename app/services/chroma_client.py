"""ChromaDB client wrapper."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChromaService:
    """Thin wrapper around the ChromaDB HTTP client."""

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        """Create the Chroma HTTP client lazily."""
        if self._client is None:
            import chromadb

            self._client = chromadb.HttpClient(host=self.host, port=self.port)
        return self._client

    def is_healthy(self) -> bool:
        """Return True when ChromaDB responds to a heartbeat."""
        try:
            self.client.heartbeat()
            return True
        except Exception as exc:
            logger.debug("ChromaDB health check failed: %s", exc)
            return False

    def get_or_create_collection(self) -> Any:
        """Return the configured collection, creating it if needed."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Internal document chunks for enterprise RAG."},
        )
