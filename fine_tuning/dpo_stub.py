"""DPO (Direct Preference Optimisation) fine-tuning — ready-to-run stub.

This stub is ready to run once 100+ preference pairs are collected from the
Gradio feedback buttons. Connect ``log_feedback()`` in app.py to write to
``data/processed/dpo_preferences.jsonl`` once a human reviewer labels good vs
bad responses (chosen vs rejected). Each line:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

It loads the LoRA-adapted Gemma model from Phase 2, layers a DPO objective on
top of the preference data, and continues training. Run on a GPU.

Run (GPU, once data exists):  ``python fine_tuning/dpo_stub.py``
"""
from __future__ import annotations

from pathlib import Path

from common.config import PROCESSED_DIR, PROJECT_ROOT, settings
from common.logging_utils import get_logger

logger = get_logger("dpo_stub")

ADAPTER_DIR = PROJECT_ROOT / "fine_tuning" / "triage_gemma_lora"
PREF_PATH = PROCESSED_DIR / "dpo_preferences.jsonl"


def run_dpo() -> None:
    if not PREF_PATH.exists():
        logger.warning("No preference data at %s — collect 100+ pairs first.", PREF_PATH)
        return

    from unsloth import FastLanguageModel
    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset as hf_load_dataset

    # 1. Load the SFT LoRA-adapted model as the policy.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR), max_seq_length=2048, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=settings.random_seed,
    )

    # 2. Preference dataset (prompt / chosen / rejected).
    dataset = hf_load_dataset("json", data_files=str(PREF_PATH), split="train")

    # 3. DPO trainer.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth/PEFT uses the adapter-free base as reference
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=100,
            learning_rate=5e-6,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            output_dir="dpo_model",
            seed=settings.random_seed,
        ),
    )
    trainer.train()
    out = PROJECT_ROOT / "fine_tuning" / "triage_gemma_dpo"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("DPO complete — saved to %s", out)


if __name__ == "__main__":
    run_dpo()
