"""Orchestrator — the single entry point for the multi-agent system.

``handle_query(query)`` runs the full pipeline:

    triage -> route to specialist (with RAG) -> escalation check
          -> hallucination check -> assemble result with latency + tokens

Everything is wrapped so that no exception ever propagates to the caller: on
failure the user gets a graceful message and the error is logged. The returned
dict is the canonical record consumed by the Gradio app and the dashboard.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

from common.config import KNOWLEDGE_BASE_DIR, settings
from common.logging_utils import get_logger, log_event
from agents.triage_agent import triage
from agents import refund_agent, technical_agent, billing_agent

logger = get_logger("orchestrator")

_SPECIALISTS = {
    "refund": refund_agent.handle,
    "technical": technical_agent.handle,
    "billing": billing_agent.handle,
}

# Keywords that should always trigger human escalation.
_HUMAN_KEYWORDS = ("human", "real person", "agent", "manager", "representative", "speak to someone")
_FRAUD_KEYWORDS = ("fraud", "hacked", "unauthorized", "unauthorised", "stolen", "didn't make", "did not make")


@lru_cache(maxsize=1)
def _escalation_matrix() -> str:
    path = KNOWLEDGE_BASE_DIR / "escalation_matrix.txt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("escalation_matrix.txt not found")
        return ""


def _extract_amount_inr(text: str) -> int | None:
    """Pull the largest INR amount mentioned in the query, if any."""
    # Matches ₹50,000 / Rs 50000 / 50000 INR / 1,00,000
    candidates = re.findall(r"(?:₹|rs\.?|inr)?\s*([\d,]{3,})", text.lower())
    amounts = []
    for c in candidates:
        try:
            amounts.append(int(c.replace(",", "")))
        except ValueError:
            continue
    return max(amounts) if amounts else None


def _check_escalation(query: str, triage_result: dict, specialist_result: dict) -> tuple[bool, list[str]]:
    """Return (escalation_required, reasons)."""
    reasons: list[str] = []
    q = query.lower()

    if any(k in q for k in _HUMAN_KEYWORDS):
        reasons.append("explicit_human_request")
    if any(k in q for k in _FRAUD_KEYWORDS):
        reasons.append("suspected_fraud")

    amount = _extract_amount_inr(query)
    if amount and amount > settings.escalation_amount_inr:
        reasons.append(f"high_value_transaction({amount})")

    if triage_result.get("urgency") == "high" and specialist_result.get("confidence") == "low":
        reasons.append("high_urgency_low_confidence")

    return (len(reasons) > 0, reasons)


def _graceful_fallback(query: str, error: str) -> dict[str, Any]:
    return {
        "query": query,
        "triage_result": {},
        "agent_used": None,
        "response": (
            "I'm sorry — something went wrong on our side. Please try again, or "
            "I can connect you with a NovaPay specialist."
        ),
        "sources_used": [],
        "confidence": "low",
        "escalation_required": True,
        "escalation_reasons": ["system_error"],
        "hallucination": {"is_hallucination": False, "severity": "none"},
        "total_latency_ms": 0.0,
        "token_counts": {"input": 0, "output": 0},
        "error": error,
    }


def handle_query(query: str, run_hallucination_check: bool = True) -> dict[str, Any]:
    """Process one customer query end to end. Never raises."""
    overall_start = time.perf_counter()
    query = (query or "").strip()

    if not query:
        return _graceful_fallback(query, "empty_query")

    try:
        # 1. Triage.
        triage_result = triage(query)
        category = triage_result["category"]

        # 2. Route to specialist (with escalation matrix as extra policy context).
        specialist_fn = _SPECIALISTS.get(category, technical_agent.handle)
        specialist_result = specialist_fn(query, triage_result, _escalation_matrix())

        response_text = specialist_result["response_text"]
        token_input = triage_result.get("input_tokens", 0) + specialist_result["input_tokens"]
        token_output = triage_result.get("output_tokens", 0) + specialist_result["output_tokens"]

        # 3. Hallucination check (guardrail).
        hallucination = {"is_hallucination": False, "severity": "none", "hallucinated_claims": []}
        if run_hallucination_check:
            try:
                from robustness.hallucination_detector import detect_hallucination

                hallucination = detect_hallucination(
                    response_text, specialist_result["retrieved_chunks"]
                )
                if hallucination.get("is_hallucination") and hallucination.get("severity") == "high":
                    log_event("hallucinations.log", query=query, claims=hallucination.get("hallucinated_claims", []))
                    response_text = (
                        "I want to make sure I give you accurate information. Let me "
                        "connect you with a specialist who can confirm the exact details."
                    )
            except Exception as exc:
                logger.warning("Hallucination check skipped: %s", exc)

        # 4. Escalation decision.
        escalation_required, reasons = _check_escalation(query, triage_result, specialist_result)
        if escalation_required:
            response_text += "\n\n(Note: this case has been flagged for human review.)"

        total_latency_ms = (time.perf_counter() - overall_start) * 1000
        return {
            "query": query,
            "triage_result": triage_result,
            "agent_used": category,
            "response": response_text,
            "sources_used": specialist_result["sources_used"],
            "confidence": specialist_result["confidence"],
            "escalation_required": escalation_required,
            "escalation_reasons": reasons,
            "hallucination": hallucination,
            "total_latency_ms": round(total_latency_ms, 1),
            "token_counts": {"input": token_input, "output": token_output},
        }
    except Exception as exc:  # final safety net
        logger.exception("Orchestrator error")
        log_event("errors.log", query=query, error=str(exc))
        return _graceful_fallback(query, str(exc))


if __name__ == "__main__":
    import json

    print(json.dumps(handle_query("My EMI of 8500 was deducted twice this month!"), indent=2, ensure_ascii=False))
