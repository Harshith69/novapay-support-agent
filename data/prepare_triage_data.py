"""Generate & curate the triage classification dataset.

Why this script exists in its current form
------------------------------------------
The original ``data/raw/triage_refund_raw.jsonl`` was a duplicate of the
billing file — it contained **zero** refund examples. Training a 3-class
classifier on 2 classes silently caps accuracy and corrupts the confusion
matrix. This script regenerates clean, balanced, de-duplicated data for all
three categories and writes a stratified train/test split.

Pipeline
--------
1. For each category, call Claude (``claude-sonnet-4-6``) in batches of 20 to
   synthesise realistic NovaPay support messages with urgency + sentiment.
2. Validate every record against the allowed label vocabulary.
3. De-duplicate on the customer message.
4. Persist per-category raw files, then a stratified 80/20 split to
   ``data/processed/{triage_train,triage_test}.jsonl``.

Run:  ``python data/prepare_triage_data.py --per-category 400``
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common.config import RAW_DIR, PROCESSED_DIR, settings
from common.llm import llm, LLMError
from common.logging_utils import get_logger
from common.parsing import extract_json, coerce_label

logger = get_logger("prepare_triage")

INSTRUCTION = "Classify the following customer query and extract urgency and sentiment"
BATCH_SIZE = 20

CATEGORY_HINTS = {
    "refund": (
        "disputed transactions, failed UPI transfers where money was debited, "
        "cashback not received, accidental double payments, refund delays"
    ),
    "technical": (
        "app login failures, biometric/fingerprint errors, OTP not received, "
        "app crashing, statement download failures, supported OS versions"
    ),
    "billing": (
        "EMI deduction errors, premium plan charges (Plus 199, Pro 499), "
        "interest calculation disputes, loan repayment schedule queries, invoices"
    ),
}

SYSTEM_PROMPT = (
    "You are generating training data for a fintech customer support triage "
    "model. Generate realistic customer queries for a digital banking app "
    "called NovaPay. Vary the phrasing, urgency, and tone. Output only a JSON "
    "array with no markdown formatting."
)


def _user_prompt(category: str) -> str:
    return (
        f"Generate {BATCH_SIZE} unique customer support messages for NovaPay in "
        f"the category '{category}' (themes: {CATEGORY_HINTS[category]}). For each, "
        "return a JSON object with fields: instruction (always set to "
        f"'{INSTRUCTION}'), input (the customer message), output (a JSON STRING "
        "with keys category, urgency, sentiment). category must be "
        f"'{category}'. urgency is one of low/medium/high. sentiment is one of "
        "neutral/frustrated/angry/calm. Mix urgency levels. Make the messages "
        "sound like real users texting support."
    )


def _validate(record: dict, category: str) -> dict | None:
    """Coerce a raw record into the canonical schema, or drop it."""
    msg = (record.get("input") or "").strip()
    if not msg:
        return None
    out = record.get("output")
    out = extract_json(out) if isinstance(out, str) else out
    if not isinstance(out, dict):
        return None

    canonical_out = {
        "category": category,  # trust the requested category, not the model
        "urgency": coerce_label(out.get("urgency"), settings.urgencies, "medium"),
        "sentiment": coerce_label(out.get("sentiment"), settings.sentiments, "neutral"),
    }
    return {
        "instruction": INSTRUCTION,
        "input": msg,
        "output": json.dumps(canonical_out, ensure_ascii=False),
    }


def generate_category(category: str, target: int) -> list[dict]:
    """Generate ``target`` validated, de-duplicated examples for one category."""
    seen: set[str] = set()
    records: list[dict] = []
    n_batches = (target + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(n_batches):
        try:
            result = llm.complete(
                system=SYSTEM_PROMPT,
                user=_user_prompt(category),
                model=settings.primary_model,
                max_tokens=2048,
                temperature=1.0,  # high diversity for synthetic data
            )
        except LLMError as exc:
            logger.error("[%s] batch %d failed: %s", category, i + 1, exc)
            continue

        parsed = extract_json(result.text)
        if not isinstance(parsed, list):
            logger.warning("[%s] batch %d returned non-list output, skipping", category, i + 1)
            continue

        for raw in parsed:
            rec = _validate(raw, category) if isinstance(raw, dict) else None
            if rec and rec["input"].lower() not in seen:
                seen.add(rec["input"].lower())
                records.append(rec)

        logger.info("[%s] batch %d/%d -> %d unique so far", category, i + 1, n_batches, len(records))
        if len(records) >= target:
            break

    return records[:target]


def stratified_split(by_category: dict[str, list[dict]], test_frac: float = 0.2):
    rng = random.Random(settings.random_seed)
    train, test = [], []
    for records in by_category.values():
        records = records[:]
        rng.shuffle(records)
        n_test = max(1, int(len(records) * test_frac))
        test.extend(records[:n_test])
        train.extend(records[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(per_category: int) -> None:
    by_category: dict[str, list[dict]] = {}
    for category in settings.categories:
        logger.info("Generating %d examples for '%s'…", per_category, category)
        recs = generate_category(category, per_category)
        by_category[category] = recs
        _write_jsonl(RAW_DIR / f"triage_{category}_raw.jsonl", recs)
        logger.info("[%s] wrote %d examples", category, len(recs))

    train, test = stratified_split(by_category)
    _write_jsonl(PROCESSED_DIR / "triage_train.jsonl", train)
    _write_jsonl(PROCESSED_DIR / "triage_test.jsonl", test)

    logger.info("Train: %d | Test: %d", len(train), len(test))
    logger.info("Train class balance: %s", dict(Counter(json.loads(r["output"])["category"] for r in train)))
    logger.info("Test  class balance: %s", dict(Counter(json.loads(r["output"])["category"] for r in test)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate NovaPay triage training data.")
    ap.add_argument("--per-category", type=int, default=400, help="examples per category")
    args = ap.parse_args()
    main(args.per_category)
