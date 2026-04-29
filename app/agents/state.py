"""State definitions for the LangGraph agent."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """Mutable state passed between LangGraph nodes."""

    question: str
    retrieved_context: str
    sources: list[str]
    answer: str
    needs_retrieval: bool

