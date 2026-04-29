"""Tests for LangGraph routing behavior."""

from __future__ import annotations

import pytest

from app.agents.graph import RagAgent


class FakeRetrievalService:
    """Retrieval fake that makes accidental retrieval obvious."""

    def retrieve(self, question: str) -> list[object]:
        raise AssertionError(f"Retrieval should not run for: {question}")


class FakeOllamaClient:
    """Generation fake for graph tests."""

    def generate(self, prompt: str) -> str:
        assert "hi, are you working?" in prompt
        return "Yes, I am working."


def test_conversational_availability_question_skips_retrieval() -> None:
    agent = RagAgent(
        retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
        ollama_client=FakeOllamaClient(),  # type: ignore[arg-type]
    )

    answer = agent.answer("hi, are you working?")

    assert answer.answer == "Yes, I am working."
    assert answer.sources == []


@pytest.mark.parametrize(
    ("question", "needs_retrieval"),
    [
        ("hi", False),
        ("hi, are you working?", False),
        ("are you online?", False),
        ("what can you do?", False),
        ("hi, what is overfitting?", True),
        ("How can I avoid overfitting?", True),
    ],
)
def test_question_classification(question: str, needs_retrieval: bool) -> None:
    agent = RagAgent(
        retrieval_service=FakeRetrievalService(),  # type: ignore[arg-type]
        ollama_client=FakeOllamaClient(),  # type: ignore[arg-type]
    )

    result = agent.classify_question_node(
        {
            "question": question,
            "retrieved_context": "",
            "sources": [],
            "answer": "",
            "needs_retrieval": True,
        }
    )

    assert result == {"needs_retrieval": needs_retrieval}
