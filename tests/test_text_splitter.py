"""Tests for the local text splitter."""

from __future__ import annotations

import pytest

from app.utils.text_splitter import split_text


def test_split_text_returns_single_chunk_for_short_text() -> None:
    chunks = split_text("Short internal note.", chunk_size=100, chunk_overlap=10)

    assert chunks == ["Short internal note."]


def test_split_text_splits_long_text() -> None:
    text = "A" * 80 + "\n\n" + "B" * 80

    chunks = split_text(text, chunk_size=90, chunk_overlap=10)

    assert len(chunks) == 2
    assert chunks[0].startswith("A")
    assert "B" in chunks[1]


def test_split_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_text("text", chunk_size=100, chunk_overlap=100)

