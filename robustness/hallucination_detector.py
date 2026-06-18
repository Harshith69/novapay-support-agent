"""Hallucination detector.

An LLM-as-fact-checker that compares a specialist's response against the exact
knowledge-base chunks it was given, and flags any claim not grounded in them.
Wired into the orchestrator so every response is checked before it reaches the
customer; a high-severity hallucination (a wrong number/timeline) suppresses
the response and triggers escalation.
"""
from __future__ import annotations

from typing import Any

from common.config import settings
from common.llm import llm, LLMError
from common.logging_utils import get_logger
from common.parsing import extract_json

logger = get_logger("hallucination_detector")

SYSTEM_PROMPT = (
    "You are a fact-checker for a fintech support AI. Given the agent's response "
    "and the exact knowledge base chunks it had access to, identify any claims in "
    "the response that are NOT supported by the chunks. Return a JSON object with: "
    "hallucinated_claims (list of strings, each a claim not in the context), "
    "is_hallucination (bool, true if list is non-empty), severity (none/low/high "
    "— high if a specific number like a refund amount or policy day count is wrong)."
)


def detect_hallucination(response: str, retrieved_chunks: list[str]) -> dict[str, Any]:
    """Return {hallucinated_claims, is_hallucination, severity}.

    On any failure returns a safe 'no hallucination detected' result so the
    guardrail never blocks a legitimate response due to checker errors.
    """
    safe_default = {"hallucinated_claims": [], "is_hallucination": False, "severity": "none"}
    if not response.strip():
        return safe_default

    context = "\n\n".join(f"[chunk {i}] {c}" for i, c in enumerate(retrieved_chunks)) or "(no context)"
    user_prompt = (
        f"Agent Response:\n{response}\n\n"
        f"Knowledge Base Chunks the agent had access to:\n{context}\n\n"
        "Identify unsupported claims and return the JSON object."
    )

    try:
        result = llm.complete(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            model=settings.primary_model,
            max_tokens=400,
            temperature=0.0,
        )
    except LLMError as exc:
        logger.error("Hallucination check failed: %s", exc)
        return safe_default

    parsed = extract_json(result.text)
    if not isinstance(parsed, dict):
        return safe_default

    claims = parsed.get("hallucinated_claims") or []
    severity = parsed.get("severity", "none")
    if severity not in ("none", "low", "high"):
        severity = "high" if claims else "none"
    return {
        "hallucinated_claims": claims,
        "is_hallucination": bool(parsed.get("is_hallucination", bool(claims))),
        "severity": severity,
    }


if __name__ == "__main__":
    import json

    demo = detect_hallucination(
        "Your refund will be processed within 1 day and you'll get ₹5000 bonus.",
        ["UPI auto-reversal happens within 3 business days. Goodwill credit is 50 INR."],
    )
    print(json.dumps(demo, indent=2, ensure_ascii=False))
