"""BERTScore evaluation against reference resolutions.

For 50 test queries (balanced across categories) we get the live agent response
from the orchestrator, find the closest reference answer in the specialist
conversation dataset (keyword-overlap match), and compute BERTScore. Reports
mean precision/recall/F1.

Run:  ``python evaluation/bertscore_eval.py``
Writes: ``evaluation/bertscore_results.json``
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from common.config import PROCESSED_DIR, EVALUATION_DIR, TRIAGE_TEST_PATH, settings
from common.logging_utils import get_logger
from agents.orchestrator import handle_query

logger = get_logger("bertscore_eval")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _load_references() -> list[tuple[str, str]]:
    """Return (category, reference_text) from specialist conversations."""
    refs = []
    for category in settings.categories:
        path = PROCESSED_DIR / f"specialist_conversations_{category}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            convo = json.loads(line)
            agent_turns = [t["text"] for t in convo.get("turns", []) if t.get("role") == "agent"]
            ref = " ".join(agent_turns) or convo.get("resolution", "")
            if ref:
                refs.append((category, ref))
    return refs


def _best_reference(query: str, category: str, refs: list[tuple[str, str]]) -> str:
    q = _tokens(query)
    best, best_overlap = "", -1
    for cat, ref in refs:
        if cat != category:
            continue
        overlap = len(q & _tokens(ref))
        if overlap > best_overlap:
            best, best_overlap = ref, overlap
    return best


def _load_test_queries(n_per_category: int = 5) -> list[tuple[str, str]]:
    if not TRIAGE_TEST_PATH.exists():
        return []
    buckets: dict[str, list[str]] = {c: [] for c in settings.categories}
    for line in TRIAGE_TEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cat = json.loads(rec["output"])["category"]
        if len(buckets[cat]) < n_per_category:
            buckets[cat].append(rec["input"])
    out = []
    for cat, qs in buckets.items():
        out.extend((q, cat) for q in qs)
    return out[:50]


def main() -> None:
    refs = _load_references()
    queries = _load_test_queries()
    if not refs or not queries:
        logger.warning("Missing specialist references or test queries — generate Phase 1 data first.")
        return

    candidates, references = [], []
    for query, category in queries:
        response = handle_query(query, run_hallucination_check=False).get("response", "")
        ref = _best_reference(query, category, refs)
        if response and ref:
            candidates.append(response)
            references.append(ref)

    from bert_score import score

    P, R, F1 = score(candidates, references, model_type="distilbert-base-uncased", verbose=False)
    results = {
        "n": len(candidates),
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }
    (EVALUATION_DIR / "bertscore_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("BERTScore:", results)


if __name__ == "__main__":
    main()
