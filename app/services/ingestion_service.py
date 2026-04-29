"""Document ingestion pipeline for ChromaDB."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.chroma_client import ChromaService
from app.utils.text_splitter import split_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of an ingestion run."""

    ingested_files: list[str]
    chunks_indexed: int
    collection: str


class IngestionService:
    """Load local documents, embed chunks, and store them in ChromaDB."""

    def __init__(
        self,
        chroma_service: ChromaService,
        docs_path: Path,
        embedding_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: Any | None = None,
    ) -> None:
        self.chroma_service = chroma_service
        self.docs_path = docs_path
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_model = embedding_model

    @property
    def embedding_model(self) -> Any:
        """Load the SentenceTransformers model on first use."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def ingest_documents(self) -> IngestionResult:
        """Ingest all markdown and text files under the configured docs path."""
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Document path does not exist: {self.docs_path}")

        files = self._discover_files()
        collection = self.chroma_service.get_or_create_collection()
        total_chunks = 0
        ingested_files: list[str] = []

        for file_path in files:
            source_name = str(file_path.relative_to(self.docs_path))
            text = file_path.read_text(encoding="utf-8")
            chunks = split_text(
                text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            if not chunks:
                logger.info("Skipping empty document: %s", source_name)
                continue

            self._delete_existing_source(collection, source_name)

            embeddings = self._encode(chunks)
            ids = [self._chunk_id(source_name, index, chunk) for index, chunk in enumerate(chunks)]
            metadatas = [
                {"source": source_name, "chunk_index": index}
                for index, _chunk in enumerate(chunks)
            ]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info("Indexed %s chunks from %s", len(chunks), source_name)
            total_chunks += len(chunks)
            ingested_files.append(source_name)

        return IngestionResult(
            ingested_files=ingested_files,
            chunks_indexed=total_chunks,
            collection=self.chroma_service.collection_name,
        )

    def _discover_files(self) -> list[Path]:
        supported_suffixes = {".md", ".txt"}
        return sorted(
            path
            for path in self.docs_path.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_suffixes
        )

    def _encode(self, chunks: list[str]) -> list[list[float]]:
        encoded = self.embedding_model.encode(chunks, normalize_embeddings=True)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [[float(value) for value in row] for row in encoded]

    @staticmethod
    def _chunk_id(source: str, index: int, chunk: str) -> str:
        digest = hashlib.sha256(f"{source}:{index}:{chunk}".encode()).hexdigest()
        return f"{source}:{index}:{digest[:16]}"

    @staticmethod
    def _delete_existing_source(collection: Any, source_name: str) -> None:
        try:
            collection.delete(where={"source": source_name})
        except Exception as exc:
            logger.debug("No existing chunks deleted for %s: %s", source_name, exc)
