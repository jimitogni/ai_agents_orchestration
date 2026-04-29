"""LangGraph orchestration for the retrieval-augmented agent."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.prompts import FALLBACK_ANSWER, build_answer_prompt
from app.agents.state import AgentState
from app.services.ollama_client import OllamaClient
from app.services.retrieval_service import RetrievalService, RetrievedChunk

CONVERSATIONAL_EXACT = {
    "hello",
    "hello there",
    "hey",
    "hey there",
    "hi",
    "hi there",
    "help",
    "thank you",
    "thanks",
}

CONVERSATIONAL_PROMPTS = (
    "are you online",
    "are you running",
    "are you there",
    "are you working",
    "can you help",
    "what can you do",
    "who are you",
)


@dataclass(frozen=True)
class AgentAnswer:
    """Final agent response."""

    answer: str
    sources: list[str]


class RagAgent:
    """LangGraph-powered RAG agent."""

    def __init__(self, retrieval_service: RetrievalService, ollama_client: OllamaClient) -> None:
        self.retrieval_service = retrieval_service
        self.ollama_client = ollama_client
        self.graph = self._build_graph()

    def answer(self, question: str) -> AgentAnswer:
        """Run the graph and return an answer with source filenames."""
        initial_state: AgentState = {
            "question": question,
            "retrieved_context": "",
            "sources": [],
            "answer": "",
            "needs_retrieval": True,
        }
        final_state = self.graph.invoke(initial_state)
        return AgentAnswer(
            answer=str(final_state.get("answer", "")),
            sources=list(final_state.get("sources", [])),
        )

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        workflow.add_node("classify_question_node", self.classify_question_node)
        workflow.add_node("retrieve_context_node", self.retrieve_context_node)
        workflow.add_node("generate_answer_node", self.generate_answer_node)
        workflow.add_node("fallback_answer_node", self.fallback_answer_node)

        workflow.set_entry_point("classify_question_node")
        workflow.add_conditional_edges(
            "classify_question_node",
            self._route_after_classification,
            {
                "retrieve": "retrieve_context_node",
                "generate": "generate_answer_node",
            },
        )
        workflow.add_conditional_edges(
            "retrieve_context_node",
            self._route_after_retrieval,
            {
                "generate": "generate_answer_node",
                "fallback": "fallback_answer_node",
            },
        )
        workflow.add_edge("generate_answer_node", END)
        workflow.add_edge("fallback_answer_node", END)
        return workflow.compile()

    def classify_question_node(self, state: AgentState) -> dict[str, bool]:
        """Decide whether the question should use document retrieval."""
        needs_retrieval = not self._is_conversational_question(state["question"])
        return {"needs_retrieval": needs_retrieval}

    def retrieve_context_node(self, state: AgentState) -> dict[str, object]:
        """Retrieve relevant document chunks from ChromaDB."""
        chunks = self.retrieval_service.retrieve(state["question"])
        context = self._format_context(chunks)
        sources = sorted({chunk.source for chunk in chunks if chunk.source})
        return {"retrieved_context": context, "sources": sources}

    def generate_answer_node(self, state: AgentState) -> dict[str, str]:
        """Generate the final answer with Ollama."""
        prompt = build_answer_prompt(
            question=state["question"],
            retrieved_context=state.get("retrieved_context", ""),
        )
        return {"answer": self.ollama_client.generate(prompt)}

    def fallback_answer_node(self, state: AgentState) -> dict[str, object]:
        """Return a grounded fallback response when no context is available."""
        return {"answer": FALLBACK_ANSWER, "sources": []}

    @staticmethod
    def _route_after_classification(state: AgentState) -> str:
        return "retrieve" if state["needs_retrieval"] else "generate"

    @staticmethod
    def _route_after_retrieval(state: AgentState) -> str:
        return "generate" if state["retrieved_context"].strip() else "fallback"

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            context_blocks.append(
                f"[Context {index}]\nSource: {chunk.source}\nContent: {chunk.text}"
            )
        return "\n\n".join(context_blocks)

    @staticmethod
    def _is_conversational_question(question: str) -> bool:
        normalized = _normalize_question(question)
        if normalized in CONVERSATIONAL_EXACT:
            return True
        if any(normalized.startswith(prompt) for prompt in CONVERSATIONAL_PROMPTS):
            return True

        greeting_prefixes = ("hello ", "hey ", "hi ")
        for greeting in greeting_prefixes:
            if normalized.startswith(greeting):
                remainder = normalized.removeprefix(greeting)
                return any(remainder.startswith(prompt) for prompt in CONVERSATIONAL_PROMPTS)

        return False


def _normalize_question(question: str) -> str:
    translation = str.maketrans({character: " " for character in string.punctuation})
    return " ".join(question.lower().translate(translation).split())
