"""Business metrics tracker.

Two responsibilities:
- ``log_interaction`` — append a normalised interaction record to the JSONL log;
- ``compute_dashboard_metrics`` — aggregate that log into the operational KPIs
  shown in the Gradio dashboard (volume, latency, escalation/hallucination
  rates, cost per ticket, and human-time saved).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from common.config import INTERACTION_LOG_PATH, settings
from common.logging_utils import append_jsonl, utc_now_iso, get_logger
import json

logger = get_logger("metrics_tracker")

# Pricing pulled from config so it stays consistent with cost accounting.
SONNET_IN, SONNET_OUT = settings.price_per_mtok["claude-sonnet-4-6"]


def log_interaction(interaction: dict[str, Any]) -> None:
    """Append one interaction record. Missing fields are filled with defaults."""
    record = {
        "timestamp": interaction.get("timestamp", utc_now_iso()),
        "query": interaction.get("query", ""),
        "category": interaction.get("category"),
        "urgency": interaction.get("urgency"),
        "agent_used": interaction.get("agent_used"),
        "latency_ms": interaction.get("latency_ms", 0),
        "token_count_input": interaction.get("token_count_input", 0),
        "token_count_output": interaction.get("token_count_output", 0),
        "confidence": interaction.get("confidence"),
        "escalation_required": bool(interaction.get("escalation_required", False)),
        "hallucination_detected": bool(interaction.get("hallucination_detected", False)),
        "user_feedback": interaction.get("user_feedback"),  # null until rated
        "feedback_note": interaction.get("feedback_note"),
    }
    append_jsonl(INTERACTION_LOG_PATH, record)


def _read_log(log_path: str | Path) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute_dashboard_metrics(log_path: str | Path = INTERACTION_LOG_PATH) -> dict[str, Any]:
    rows = _read_log(log_path)
    total = len(rows)
    if total == 0:
        return {"total_queries": 0, "by_category": {}, "message": "No interactions logged yet."}

    by_cat_count: dict[str, int] = defaultdict(int)
    by_cat_latency: dict[str, list[float]] = defaultdict(list)
    by_cat_esc: dict[str, int] = defaultdict(int)
    by_cat_hallu: dict[str, int] = defaultdict(int)
    by_cat_conf: dict[str, list[float]] = defaultdict(list)

    conf_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    total_in = total_out = 0
    total_latency = 0.0
    escalations = hallucinations = 0

    for r in rows:
        cat = r.get("category") or "unknown"
        by_cat_count[cat] += 1
        by_cat_latency[cat].append(r.get("latency_ms", 0))
        by_cat_esc[cat] += int(r.get("escalation_required", False))
        by_cat_hallu[cat] += int(r.get("hallucination_detected", False))
        by_cat_conf[cat].append(conf_map.get(r.get("confidence"), 0.0))

        total_in += r.get("token_count_input", 0)
        total_out += r.get("token_count_output", 0)
        total_latency += r.get("latency_ms", 0)
        escalations += int(r.get("escalation_required", False))
        hallucinations += int(r.get("hallucination_detected", False))

    avg_latency_ms = total_latency / total
    total_cost = (total_in / 1e6) * SONNET_IN + (total_out / 1e6) * SONNET_OUT

    # Human time saved: baseline minutes minus actual handling time, summed.
    actual_minutes = sum(r.get("latency_ms", 0) for r in rows) / 1000 / 60
    baseline_minutes = settings.baseline_human_minutes_per_ticket * total
    minutes_saved = max(0.0, baseline_minutes - actual_minutes)

    by_category = {}
    for cat, count in by_cat_count.items():
        lats = by_cat_latency[cat]
        confs = by_cat_conf[cat]
        by_category[cat] = {
            "count": count,
            "pct": round(100 * count / total, 1),
            "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
            "escalation_rate_pct": round(100 * by_cat_esc[cat] / count, 1),
            "hallucination_rate_pct": round(100 * by_cat_hallu[cat] / count, 1),
            "avg_confidence": round(sum(confs) / len(confs), 2) if confs else 0,
        }

    return {
        "total_queries": total,
        "by_category": by_category,
        "avg_latency_ms": round(avg_latency_ms, 1),
        "escalation_rate_pct": round(100 * escalations / total, 1),
        "hallucination_rate_pct": round(100 * hallucinations / total, 1),
        "avg_confidence": round(
            sum(conf_map.get(r.get("confidence"), 0.0) for r in rows) / total, 2
        ),
        "total_tokens": {"input": total_in, "output": total_out},
        "estimated_total_cost_usd": round(total_cost, 4),
        "estimated_cost_per_ticket_usd": round(total_cost / total, 5),
        "human_minutes_saved": round(minutes_saved, 1),
    }


if __name__ == "__main__":
    print(json.dumps(compute_dashboard_metrics(), indent=2, ensure_ascii=False))
