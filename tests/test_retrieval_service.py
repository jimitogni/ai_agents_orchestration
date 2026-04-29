"""Tests for retrieval result parsing and ChromaDB calls."""

from __future__ import annotations

from typing import Any

from app.services.retrieval_service import RetrievalService


class FakeEmbeddingModel:
    """Small fake embedding model for unit tests."""

    def encode(self, values: list[str], normalize_embeddings: bool) -> list[list[float]]:
        assert values == ["What is overfitting?"]
        assert normalize_embeddings is True
        return [[0.1, 0.2, 0.3]]


class FakeCollection:
    """Small fake Chroma collection for unit tests."""

    def query(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]
        assert kwargs["n_results"] == 2
        return {
            "documents": [["Overfitting is a generalization problem."]],
            "metadatas": [[{"source": "machine_learning_notes.md"}]],
            "distances": [[0.12]],
        }


class FakeChromaService:
    """Small fake Chroma service for unit tests."""

    def get_or_create_collection(self) -> FakeCollection:
        return FakeCollection()


def test_retrieve_returns_chunks_with_sources() -> None:
    service = RetrievalService(
        chroma_service=FakeChromaService(),  # type: ignore[arg-type]
        embedding_model_name="fake-model",
        top_k=2,
        embedding_model=FakeEmbeddingModel(),
    )

    results = service.retrieve("What is overfitting?")

    assert len(results) == 1
    assert results[0].text == "Overfitting is a generalization problem."
    assert results[0].source == "machine_learning_notes.md"
    assert results[0].score == 0.12

