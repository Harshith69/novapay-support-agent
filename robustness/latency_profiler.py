"""Latency profiler.

Runs a representative query mix through the full orchestrator and reports a
per-stage and end-to-end latency distribution (mean/median/p95/p99). If p95
total latency breaches the budget it prints concrete optimisation levers.

Run:  ``python robustness/latency_profiler.py``
Writes: ``robustness/latency_results.json``
"""
from __future__ import annotations

import json
import statistics
import time

from common.config import ROBUSTNESS_DIR, settings
from common.logging_utils import get_logger
from agents import refund_agent, technical_agent, billing_agent
from agents.triage_agent import triage
from rag.retriever import retrieve
from robustness.hallucination_detector import detect_hallucination

logger = get_logger("latency_profiler")

QUERIES = [
    # refund
    # 2 per category + 2 edge = 8 queries (token-budget-friendly)
    "My UPI payment failed but money was debited", "I was charged twice for one payment",
    "I'm not getting the OTP", "Fingerprint login stopped working",
    "EMI deducted twice this month", "Why am I charged 199 for Plus?",
    "", "I need to transfer 100000 INR refund urgently human please",
]

_SPECIALISTS = {"refund": refund_agent, "technical": technical_agent, "billing": billing_agent}


def _profile_one(query: str) -> dict:
    t0 = time.perf_counter()
    tri = triage(query)
    t1 = time.perf_counter()
    chunks = retrieve(query)
    t2 = time.perf_counter()
    specialist = _SPECIALISTS.get(tri["category"], technical_agent)
    res = specialist.handle(query, tri)
    t3 = time.perf_counter()
    detect_hallucination(res["response_text"], res["retrieved_chunks"])
    t4 = time.perf_counter()
    return {
        "query": query[:40],
        "triage_latency_ms": round((t1 - t0) * 1000, 1),
        "retrieval_latency_ms": round((t2 - t1) * 1000, 1),
        "agent_llm_latency_ms": round((t3 - t2) * 1000, 1),
        "hallucination_check_latency_ms": round((t4 - t3) * 1000, 1),
        "total_latency_ms": round((t4 - t0) * 1000, 1),
    }


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[k]


def main() -> None:
    rows = [_profile_one(q) for q in QUERIES]
    totals = [r["total_latency_ms"] for r in rows]
    stats = {
        "mean_ms": round(statistics.mean(totals), 1),
        "median_ms": round(statistics.median(totals), 1),
        "p95_ms": round(_pct(totals, 95), 1),
        "p99_ms": round(_pct(totals, 99), 1),
    }
    out = {"stats": stats, "rows": rows}
    (ROBUSTNESS_DIR / "latency_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Latency summary (ms) ===")
    for k, v in stats.items():
        print(f"  {k:10s}: {v}")
    if stats["p95_ms"] > settings.total_latency_budget_ms:
        print("\n⚠ p95 exceeds budget. Optimisation suggestions:")
        print("  - cache ChromaDB embeddings / keep the collection warm")
        print("  - reduce top_k from 3 to 2")
        print("  - run retrieval and the LLM call concurrently (async)")


if __name__ == "__main__":
    main()
