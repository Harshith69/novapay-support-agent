"""Generate multi-turn specialist conversations (few-shot reference data).

These conversations are NOT used to fine-tune anything. They serve two roles:
- few-shot style references the specialist agents can draw tone from, and
- a pool of reference answers for the BERTScore evaluation in Phase 5.

Each conversation has 3–5 turns and ends in a resolution. Output:
``data/processed/specialist_conversations_{type}.jsonl`` — one JSON object per
line with fields: id, category, turns (list of {role, text}), resolution.

Run:  ``python data/prepare_specialist_data.py --per-type 300``
"""
from __future__ import annotations

import argparse
import json

from common.config import PROCESSED_DIR, settings
from common.llm import llm, LLMError
from common.logging_utils import get_logger
from common.parsing import extract_json

logger = get_logger("prepare_specialist")

BATCH_SIZE = 10  # conversations per API call

SYSTEM_PROMPT = (
    "You generate realistic customer-support conversations for NovaPay, a "
    "digital banking app. Each conversation must stay strictly within NovaPay "
    "policy and end with a clear resolution. Output only a JSON array, no markdown."
)

TYPE_BRIEF = {
    "refund": "disputed transactions, failed UPI debits, cashback, double payments. "
              "Refunds: UPI within 3 business days, disputes 7-day investigation, "
              "escalate if delayed beyond 10 days.",
    "technical": "login/biometric/OTP issues, crashes, statement downloads. "
                 "OTP resend after 60s, max 3 attempts then email OTP; biometric reset steps.",
    "billing": "EMI errors, premium plans (Plus 199/mo, Pro 499/mo), invoices, "
               "loan schedules. Double EMI is auto-reversed within 5 business days.",
}


def _user_prompt(agent_type: str) -> str:
    return (
        f"Generate {BATCH_SIZE} unique NovaPay support conversations of type "
        f"'{agent_type}' (themes: {TYPE_BRIEF[agent_type]}). Each conversation: "
        "3 to 5 alternating turns starting with the customer. Return a JSON array "
        "where each item is an object with fields: turns (array of objects with "
        "'role' = 'customer' or 'agent' and 'text'), and resolution (one sentence "
        "describing the outcome). The agent must follow NovaPay policy and never "
        "invent amounts or timelines beyond the brief."
    )


def generate_type(agent_type: str, target: int) -> list[dict]:
    convos: list[dict] = []
    n_batches = (target + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batches):
        try:
            result = llm.complete(
                system=SYSTEM_PROMPT,
                user=_user_prompt(agent_type),
                model=settings.fallback_model,
                max_tokens=4096,
                temperature=0.9,
            )
        except LLMError as exc:
            logger.error("[%s] batch %d failed: %s", agent_type, i + 1, exc)
            continue

        parsed = extract_json(result.text)
        if not isinstance(parsed, list):
            logger.warning("[%s] batch %d non-list, skipping", agent_type, i + 1)
            continue

        for item in parsed:
            turns = item.get("turns") if isinstance(item, dict) else None
            if isinstance(turns, list) and 2 <= len(turns) <= 6:
                convos.append({
                    "id": f"{agent_type}-{len(convos):04d}",
                    "category": agent_type,
                    "turns": turns,
                    "resolution": item.get("resolution", ""),
                })
        logger.info("[%s] batch %d/%d -> %d convos", agent_type, i + 1, n_batches, len(convos))
        if len(convos) >= target:
            break
    return convos[:target]


def main(per_type: int) -> None:
    for agent_type in settings.categories:
        convos = generate_type(agent_type, per_type)
        path = PROCESSED_DIR / f"specialist_conversations_{agent_type}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for c in convos:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        logger.info("[%s] wrote %d conversations -> %s", agent_type, len(convos), path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate NovaPay specialist conversations.")
    ap.add_argument("--per-type", type=int, default=300, help="conversations per agent type")
    args = ap.parse_args()
    main(args.per_type)
