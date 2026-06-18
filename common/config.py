"""Central configuration for NovaPay.

All paths, model identifiers, thresholds and pricing live here so that a
reviewer can understand the whole system's wiring from a single file, and so
that nothing is hard-coded across two dozen modules.

Design choices worth noting:
- Paths are resolved relative to the project root via ``pathlib`` and created
  on import, so no script ever crashes on a missing directory.
- Secrets are loaded from a ``.env`` file (via ``python-dotenv``) if present,
  then from the process environment. Never hard-code keys.
- A frozen dataclass gives us autocomplete + immutability without a heavy
  settings framework.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Windows consoles default to cp1252 and crash on ₹/emoji in our data. Force
# UTF-8 on stdout/stderr once, here, so every script is safe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

try:  # optional dependency; the system still runs if it is absent
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

CHROMA_DIR = PROJECT_ROOT / "rag" / "chroma_store"
LOGS_DIR = PROJECT_ROOT / "logs"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
ROBUSTNESS_DIR = PROJECT_ROOT / "robustness"

# Files
TRIAGE_TRAIN_PATH = PROCESSED_DIR / "triage_train.jsonl"
TRIAGE_TEST_PATH = PROCESSED_DIR / "triage_test.jsonl"
INTERACTION_LOG_PATH = LOGS_DIR / "interaction_log.jsonl"

_ALL_DIRS = [
    DATA_DIR, RAW_DIR, PROCESSED_DIR, KNOWLEDGE_BASE_DIR,
    LOGS_DIR, EVALUATION_DIR, ROBUSTNESS_DIR,
]
for _d in _ALL_DIRS:
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Immutable system settings. Import the singleton ``settings`` below."""

    # --- LLM provider -----------------------------------------------------
    # Which hosted-LLM backend common/llm.py talks to. One of:
    # "groq" (free), "anthropic" (paid), "gemini" (free), "ollama" (local).
    # Overridable at runtime via the LLM_PROVIDER env var.
    llm_provider: str = os.environ.get("LLM_PROVIDER", "groq")

    # --- Models -----------------------------------------------------------
    # Strong model: specialist agents + evaluation/judging.
    # Fast model: fallbacks and cheap baselines.
    # Defaults below are Groq free-tier models; see model_for() for per-provider
    # resolution so we never hard-code a vendor's id at the call sites.
    primary_model: str = "llama-3.3-70b-versatile"
    fallback_model: str = "llama-3.1-8b-instant"
    # Fine-tuned triage classifier (LoRA adapter on HF Hub).
    triage_base_model: str = "unsloth/gemma-3-4b-it"
    triage_hub_repo: str = "Harshith69/novapay-triage-gemma"
    # Embeddings for RAG — free, CPU-friendly.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Per-provider model ids for (strong, fast). Lets us switch providers
    # without touching any agent or evaluation code.
    provider_models: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "groq": ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
            "anthropic": ("claude-sonnet-4-6", "claude-haiku-4-5"),
            "gemini": ("gemini-2.0-flash", "gemini-2.0-flash-lite"),
            "ollama": ("llama3.1:8b", "llama3.2:3b"),
        }
    )

    # --- Categories / labels ---------------------------------------------
    categories: tuple[str, ...] = ("refund", "technical", "billing")
    urgencies: tuple[str, ...] = ("low", "medium", "high")
    sentiments: tuple[str, ...] = ("neutral", "frustrated", "angry", "calm")

    # --- RAG --------------------------------------------------------------
    chroma_collection: str = "novapay_kb"
    chunk_size: int = 300
    chunk_overlap: int = 50
    default_top_k: int = 3

    # --- Confidence thresholds (mean relevance score) --------------------
    confidence_high: float = 0.75
    confidence_medium: float = 0.50

    # --- Escalation rules -------------------------------------------------
    escalation_amount_inr: int = 50_000
    max_agent_attempts: int = 2

    # --- Latency budgets (ms) --------------------------------------------
    triage_latency_budget_ms: int = 3_000
    total_latency_budget_ms: int = 8_000

    # --- API resilience ---------------------------------------------------
    api_max_retries: int = 4
    api_base_delay_s: float = 1.0
    api_max_tokens: int = 1_024
    request_timeout_s: float = 60.0

    # --- Pricing (USD per 1M tokens) for the cost dashboard --------------
    # Groq free-tier models are $0; Anthropic kept for the paid fallback so the
    # dashboard still computes a realistic figure if you switch providers.
    price_per_mtok: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            # model: (input_price, output_price)
            "llama-3.3-70b-versatile": (0.0, 0.0),
            "llama-3.1-8b-instant": (0.0, 0.0),
            "gemini-2.0-flash": (0.0, 0.0),
            "gemini-2.0-flash-lite": (0.0, 0.0),
            "claude-sonnet-4-6": (3.0, 15.0),
            "claude-haiku-4-5": (1.0, 5.0),
        }
    )

    # --- Misc -------------------------------------------------------------
    random_seed: int = 42
    baseline_human_minutes_per_ticket: float = 8.0

    def __post_init__(self) -> None:
        # Resolve strong/fast model ids from the active provider so call sites
        # (agents, judge, data gen) never name a vendor model directly.
        if self.llm_provider in self.provider_models:
            strong, fast = self.provider_models[self.llm_provider]
            object.__setattr__(self, "primary_model", strong)
            object.__setattr__(self, "fallback_model", fast)

    @property
    def use_finetuned_triage(self) -> bool:
        """Load the fine-tuned Gemma classifier only when explicitly enabled.

        Default off so CPU/local runs go straight to the Claude fallback instead
        of pulling a 9B model over the network. Set USE_FINETUNED_TRIAGE=1 on
        GPU/Spaces where the adapter is available.
        """
        return os.environ.get("USE_FINETUNED_TRIAGE", "0").lower() in ("1", "true", "yes")

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")

    @property
    def groq_api_key(self) -> str | None:
        return os.environ.get("GROQ_API_KEY")

    @property
    def gemini_api_key(self) -> str | None:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    @property
    def hf_token(self) -> str | None:
        return os.environ.get("HF_TOKEN")

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for a single call from token usage."""
        in_price, out_price = self.price_per_mtok.get(model, (0.0, 0.0))
        return (input_tokens / 1e6) * in_price + (output_tokens / 1e6) * out_price


settings = Settings()
