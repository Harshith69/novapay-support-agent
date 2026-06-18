"""Fine-tune the Gemma triage classifier with Unsloth + LoRA.

Designed to run on a free Colab T4 (16GB) or Kaggle P100 — NOT on a CPU box.
On <16GB VRAM, set ``--batch-size 1 --grad-accum 8``.

Pipeline: load 4-bit Gemma -> attach LoRA adapters -> SFT on the Alpaca-format
triage data -> save adapter + tokenizer locally -> push to the HF Hub.

Run (on GPU):  ``python fine_tuning/train_triage.py --max-steps 200``
"""
from __future__ import annotations

import argparse

from common.config import PROJECT_ROOT, settings
from common.logging_utils import get_logger
from fine_tuning.prepare_data import load_dataset

logger = get_logger("train_triage")

ADAPTER_DIR = PROJECT_ROOT / "fine_tuning" / "triage_gemma_lora"
MAX_SEQ_LEN = 2048


def train(max_steps: int, batch_size: int, grad_accum: int, push: bool) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # 1. Load base model in 4-bit.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings.triage_base_model,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )

    # 2. Attach LoRA adapters.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=settings.random_seed,
    )

    # 3. Data.
    train_ds = load_dataset("train")

    # 4. Trainer.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=20,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            seed=settings.random_seed,
            output_dir="triage_model",
        ),
    )

    logger.info("Starting training for %d steps…", max_steps)
    trainer.train()

    # 5. Save adapter + tokenizer.
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    logger.info("Saved LoRA adapter to %s", ADAPTER_DIR)

    # 6. Push to Hub.
    if push:
        if not settings.hf_token:
            logger.warning("HF_TOKEN not set — skipping push to Hub.")
        else:
            model.push_to_hub(settings.triage_hub_repo, token=settings.hf_token)
            tokenizer.push_to_hub(settings.triage_hub_repo, token=settings.hf_token)
            logger.info("Pushed adapter to HF Hub repo '%s'", settings.triage_hub_repo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune NovaPay triage model (GPU only).")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--no-push", action="store_true", help="don't push to HF Hub")
    args = ap.parse_args()
    train(args.max_steps, args.batch_size, args.grad_accum, push=not args.no_push)
