"""Prompt templates used by the RAG agent."""

from __future__ import annotations


def build_answer_prompt(question: str, retrieved_context: str) -> str:
    """Build the prompt sent to the local LLM."""
    if retrieved_context.strip():
        return f"""You are an enterprise knowledge assistant.
Answer the question using only the provided internal context.
If the context is incomplete, say what is missing instead of inventing details.
Keep the answer concise and practical.

Internal context:
{retrieved_context}

Question:
{question}

Answer:"""

    return f"""You are an enterprise knowledge assistant.
Answer the user briefly and clearly.
If the user asks about internal documents, explain that document ingestion or retrieval is needed.

Question:
{question}

Answer:"""


FALLBACK_ANSWER = (
    "I could not find relevant internal context for that question. "
    "Try ingesting documents first, or ask a question that is covered by the files in data/docs."
)

