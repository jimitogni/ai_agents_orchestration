"""Simple text splitter used by the ingestion pipeline."""

from __future__ import annotations


def split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks while preserving paragraph boundaries when possible."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(paragraph, chunk_size, chunk_overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
        current = paragraph

    if current:
        chunks.append(current.strip())

    return _add_overlap(chunks, chunk_overlap, chunk_size)


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def _add_overlap(chunks: list[str], chunk_overlap: int, chunk_size: int) -> list[str]:
    if chunk_overlap == 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:], strict=False):
        prefix = previous[-chunk_overlap:].strip()
        combined = f"{prefix}\n\n{current}".strip()
        if len(combined) > chunk_size:
            combined = combined[-chunk_size:].strip()
        overlapped.append(combined)
    return overlapped
