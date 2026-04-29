"""Retrieval service for querying ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.chroma_client import ChromaService


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk returned by vector search."""

    text: str
    source: str
    score: float | None = None


class RetrievalService:
    """Embed a user question and retrieve relevant chunks from ChromaDB."""

    def __init__(
        self,
        chroma_service: ChromaService,
        embedding_model_name: str,
        top_k: int,
        embedding_model: Any | None = None,
    ) -> None:
        self.chroma_service = chroma_service
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self._embedding_model = embedding_model

    @property
    def embedding_model(self) -> Any:
        """Load the SentenceTransformers model on first use."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return relevant chunks for a question."""
        if not question.strip():
            return []

        collection = self.chroma_service.get_or_create_collection()
        query_embedding = self._encode_query(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._parse_results(results)

    def _encode_query(self, question: str) -> list[float]:
        encoded = self.embedding_model.encode([question], normalize_embeddings=True)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [float(value) for value in encoded[0]]

    @staticmethod
    def _parse_results(results: dict[str, Any]) -> list[RetrievedChunk]:
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            distance = distances[index] if index < len(distances) else None
            source = str(metadata.get("source", "unknown"))
            score = None if distance is None else float(distance)
            chunks.append(RetrievedChunk(text=str(document), source=source, score=score))
        return chunks

