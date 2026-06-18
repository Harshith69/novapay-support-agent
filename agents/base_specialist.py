"""Shared specialist-agent implementation.

The refund, technical and billing agents are identical in mechanics — they
differ only in their system prompt (persona + policy guardrails). Rather than
copy-paste the RAG + LLM + confidence logic three times, that logic lives here
once and each specialist is a thin :class:`SpecialistAgent` instance.

Contract: ``handle(query, triage_result, extra_context)`` returns
``{response_text, sources_used, confidence, retrieved_chunks, input_tokens,
output_tokens, latency_ms}`` — everything the orchestrator needs for routing,
hallucination checks, cost accounting and the dashboard.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from common.config import settings
from common.llm import llm, LLMError
from common.logging_utils import get_logger
from rag.retriever import retrieve, format_context

logger = get_logger("specialist")


def _confidence(scores: list[float]) -> str:
    if not scores:
        return "low"
    avg = sum(scores) / len(scores)
    if avg > settings.confidence_high:
        return "high"
    if avg > settings.confidence_medium:
        return "medium"
    return "low"


@dataclass
class SpecialistAgent:
    """A RAG-grounded support specialist driven by a system prompt."""

    name: str
    system_prompt: str

    def handle(
        self,
        query: str,
        triage_result: dict | None = None,
        extra_context: str = "",
        top_k: int = settings.default_top_k,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        triage_result = triage_result or {}
        urgency = triage_result.get("urgency", "medium")
        sentiment = triage_result.get("sentiment", "neutral")

        # 1. Retrieve grounding context.
        chunks = retrieve(query, top_k=top_k)
        context = format_context(chunks)
        scores = [c["relevance_score"] for c in chunks]
        sources = sorted({c["source"] for c in chunks})

        # 2. Build the grounded prompt.
        user_prompt = (
            f"NovaPay policy context:\n{context}\n\n"
            f"{('Additional policy: ' + extra_context) if extra_context else ''}\n\n"
            f"Customer query: {query}\n"
            f"Customer urgency: {urgency}. Customer sentiment: {sentiment}.\n\n"
            "Respond to the customer now, using ONLY the policy context above."
        )

        # 3. Generate (graceful fallback on failure).
        try:
            result = llm.complete(
                system=self.system_prompt,
                user=user_prompt,
                model=settings.primary_model,
                max_tokens=settings.api_max_tokens,
                temperature=0.4,
            )
            response_text = result.text
            in_tok, out_tok = result.input_tokens, result.output_tokens
        except LLMError as exc:
            logger.error("[%s] generation failed: %s", self.name, exc)
            response_text = (
                "I'm sorry, I'm having trouble accessing our systems right now. "
                "Let me connect you with a NovaPay specialist who can help."
            )
            in_tok = out_tok = 0

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "agent": self.name,
            "response_text": response_text,
            "sources_used": sources,
            "confidence": _confidence(scores),
            "retrieved_chunks": [c["text"] for c in chunks],
            "relevance_scores": scores,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": round(latency_ms, 1),
        }
