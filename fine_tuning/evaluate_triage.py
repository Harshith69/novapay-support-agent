"""Evaluate the fine-tuned triage model on the held-out test set.

Computes overall accuracy, per-class precision/recall/F1, a confusion matrix
(saved to JSON), and urgency accuracy. Also runs Claude Haiku as a baseline and
prints a side-by-side accuracy table.

Run (GPU for the Gemma path):  ``python fine_tuning/evaluate_triage.py``
"""
from __future__ import annotations

import json

from common.config import EVALUATION_DIR, TRIAGE_TEST_PATH, settings
from common.llm import llm, LLMError
from common.logging_utils import get_logger
from common.parsing import extract_json, coerce_label
from fine_tuning.prepare_data import format_example

logger = get_logger("evaluate_triage")


def _load_test() -> list[dict]:
    return [json.loads(l) for l in TRIAGE_TEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _gold(record: dict) -> dict:
    return json.loads(record["output"])


# --- Predictors --------------------------------------------------------------
def _predict_gemma(records: list[dict]) -> list[dict]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(settings.triage_hub_repo), max_seq_length=2048, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    preds = []
    for r in records:
        prompt = format_example(r, include_output=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = extract_json(text) or {}
        preds.append(_normalise(parsed))
    return preds


def _predict_claude(records: list[dict], model: str) -> list[dict]:
    system = (
        "Classify the NovaPay customer query. Return ONLY JSON with keys category "
        "(refund/technical/billing), urgency (low/medium/high), sentiment "
        "(neutral/frustrated/angry/calm)."
    )
    preds = []
    for r in records:
        try:
            res = llm.complete(system=system, user=r["input"], model=model, max_tokens=120, temperature=0.0)
            preds.append(_normalise(extract_json(res.text) or {}))
        except LLMError:
            preds.append(_normalise({}))
    return preds


def _normalise(parsed: dict) -> dict:
    return {
        "category": coerce_label(parsed.get("category"), settings.categories, "technical"),
        "urgency": coerce_label(parsed.get("urgency"), settings.urgencies, "medium"),
    }


# --- Metrics -----------------------------------------------------------------
def _metrics(gold: list[dict], pred: list[dict]) -> dict:
    cats = settings.categories
    cat_index = {c: i for i, c in enumerate(cats)}
    confusion = [[0] * len(cats) for _ in cats]
    correct = urg_correct = 0

    for g, p in zip(gold, pred):
        confusion[cat_index[g["category"]]][cat_index[p["category"]]] += 1
        correct += int(g["category"] == p["category"])
        urg_correct += int(g["urgency"] == p["urgency"])

    n = len(gold)
    per_class = {}
    for c in cats:
        i = cat_index[c]
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(len(cats))) - tp
        fn = sum(confusion[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

    return {
        "accuracy": round(correct / n, 4),
        "urgency_accuracy": round(urg_correct / n, 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(cats), "matrix": confusion},
    }


def _print_confusion(cm: dict) -> None:
    labels, matrix = cm["labels"], cm["matrix"]
    header = "actual\\pred | " + " | ".join(f"{l[:7]:>7}" for l in labels)
    print(header)
    print("-" * len(header))
    for label, row in zip(labels, matrix):
        print(f"{label[:11]:<11} | " + " | ".join(f"{v:>7}" for v in row))


def main() -> None:
    records = _load_test()
    gold = [_normalise(_gold(r)) for r in records]

    # Fine-tuned Gemma (may be unavailable on CPU — handle gracefully).
    try:
        gemma_pred = _predict_gemma(records)
        gemma_metrics = _metrics(gold, gemma_pred)
    except Exception as exc:
        logger.warning("Gemma evaluation skipped (%s).", exc)
        gemma_metrics = None

    haiku_pred = _predict_claude(records, settings.fallback_model)
    haiku_metrics = _metrics(gold, haiku_pred)

    print("\n=== Accuracy comparison ===")
    if gemma_metrics:
        print(f"  Fine-Tuned Gemma : {gemma_metrics['accuracy']:.3f}")
    print(f"  Claude Haiku     : {haiku_metrics['accuracy']:.3f}")

    if gemma_metrics:
        print("\n=== Fine-Tuned Gemma per-class ===")
        for c, m in gemma_metrics["per_class"].items():
            print(f"  {c:10s} P={m['precision']} R={m['recall']} F1={m['f1']}")
        print(f"\nUrgency accuracy: {gemma_metrics['urgency_accuracy']:.3f}")
        print("\nConfusion matrix:")
        _print_confusion(gemma_metrics["confusion_matrix"])
        (EVALUATION_DIR / "confusion_matrix.json").write_text(
            json.dumps(gemma_metrics["confusion_matrix"], indent=2), encoding="utf-8"
        )
        acc = gemma_metrics["accuracy"]
        if acc < 0.85:
            logger.warning("Accuracy %.3f below 0.85 — increase max_steps to 300.", acc)
        elif acc < 0.88:
            logger.info("Accuracy %.3f below 0.88 target but acceptable.", acc)


if __name__ == "__main__":
    main()
