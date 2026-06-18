"""A/B/C system comparison.

Compares three systems on 30 test queries using the LLM judge:
  A) full system: (fine-tuned triage routing) + RAG specialist agents
  B) vanilla model, no RAG, no routing (single generic prompt)
  C) Claude Haiku standalone, single-turn, system prompt + query only

Produces a per-query score table and mean scores per system.

Run:  ``python evaluation/ab_comparison.py``
Writes: ``evaluation/ab_results.json``
"""
from __future__ import annotations

import json

from common.config import EVALUATION_DIR, TRIAGE_TEST_PATH, settings
from common.llm import llm, LLMError
from common.logging_utils import get_logger
from agents.orchestrator import handle_query
from evaluation.llm_judge import judge_response

logger = get_logger("ab_comparison")

GENERIC_SYSTEM = "You are a helpful customer support assistant for a banking app."


def _avg_score(verdict: dict) -> float:
    keys = ("faithfulness", "helpfulness", "tone")
    vals = [float(verdict.get(k, 0) or 0) for k in keys]
    return round(sum(vals) / len(vals), 3)


def _system_b(query: str) -> str:
    """Vanilla model, no RAG, no routing."""
    try:
        return llm.complete(system=GENERIC_SYSTEM, user=query,
                            model=settings.fallback_model, max_tokens=400, temperature=0.4).text
    except LLMError:
        return ""


def _system_c(query: str) -> str:
    """Claude Haiku standalone with a light support persona."""
    try:
        return llm.complete(
            system="You are NovaPay support. Be concise, empathetic and policy-safe.",
            user=query, model=settings.fallback_model, max_tokens=400, temperature=0.4,
        ).text
    except LLMError:
        return ""


def _load_queries(n: int = 10) -> list[str]:
    if not TRIAGE_TEST_PATH.exists():
        return []
    qs = []
    for line in TRIAGE_TEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qs.append(json.loads(line)["input"])
        if len(qs) >= n:
            break
    return qs


def main() -> None:
    queries = _load_queries()
    if not queries:
        logger.warning("No test queries found — run Phase 1 data generation first.")
        return

    # Retrieve the KB context once per query so ALL three systems are judged
    # against the same ground truth. Without this, B and C get perfect
    # faithfulness scores because the judge has nothing to check against.
    from rag.retriever import retrieve, format_context

    rows = []
    for query in queries:
        a = handle_query(query, run_hallucination_check=False)
        resp_a = a.get("response", "")
        # Ground-truth chunks the judge uses for ALL three systems.
        chunks = retrieve(query, top_k=3)
        kb_context = [c["text"] for c in chunks]
        resp_b, resp_c = _system_b(query), _system_c(query)

        score_a = _avg_score(judge_response(query, resp_a, kb_context))
        score_b = _avg_score(judge_response(query, resp_b, kb_context))
        score_c = _avg_score(judge_response(query, resp_c, kb_context))
        rows.append({
            "query": query, "system_a_score": score_a,
            "system_b_score": score_b, "system_c_score": score_c,
        })
        logger.info("A=%.2f B=%.2f C=%.2f | %s", score_a, score_b, score_c, query[:40])

    means = {
        "system_a_mean": round(sum(r["system_a_score"] for r in rows) / len(rows), 3),
        "system_b_mean": round(sum(r["system_b_score"] for r in rows) / len(rows), 3),
        "system_c_mean": round(sum(r["system_c_score"] for r in rows) / len(rows), 3),
    }
    (EVALUATION_DIR / "ab_results.json").write_text(
        json.dumps({"means": means, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== Mean scores ===")
    print(f"  A (full system + RAG): {means['system_a_mean']}")
    print(f"  B (vanilla, no RAG)  : {means['system_b_mean']}")
    print(f"  C (Haiku standalone) : {means['system_c_mean']}")


if __name__ == "__main__":
    main()
