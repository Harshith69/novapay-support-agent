"""Format the processed triage data into the training prompt template.

Shared by the trainer and evaluator so both use an identical prompt format.
Returns a HuggingFace ``Dataset`` when ``datasets`` is installed, else a list
of dicts (useful for inspection on a machine without the training stack).
"""
from __future__ import annotations

import json
from pathlib import Path

from common.config import TRIAGE_TRAIN_PATH, TRIAGE_TEST_PATH

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

# Prompt without the answer — used at inference/eval time.
INFERENCE_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run data/prepare_triage_data.py first.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def format_example(record: dict, include_output: bool = True) -> str:
    tmpl = ALPACA_TEMPLATE if include_output else INFERENCE_TEMPLATE
    return tmpl.format(
        instruction=record["instruction"], input=record["input"],
        output=record.get("output", "") if include_output else "",
    )


def load_dataset(split: str = "train"):
    path = TRIAGE_TRAIN_PATH if split == "train" else TRIAGE_TEST_PATH
    records = _read_jsonl(path)
    formatted = [{"text": format_example(r, include_output=True), **r} for r in records]
    try:
        from datasets import Dataset

        return Dataset.from_list(formatted)
    except ImportError:
        return formatted


if __name__ == "__main__":
    ds = load_dataset("train")
    print(f"Loaded {len(ds)} training examples.")
    print(ds[0]["text"] if hasattr(ds, "__getitem__") else ds[0])
