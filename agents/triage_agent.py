"""Triage agent.

Classifies a raw customer query into {category, urgency, sentiment} plus a
short summary. Primary path is the fine-tuned Gemma classifier loaded from the
HF Hub; if the model is unavailable or its output cannot be parsed, it falls
back to ``claude-haiku-4-5`` with a strict-JSON system prompt. Every fallback
is logged so we can measure how often the fine-tuned model is trusted.

Design notes
------------
- The Gemma model is loaded lazily and cached; on a CPU-only box (or when
  transformers/torch are absent) loading simply fails and we use the fallback.
  This keeps the whole system runnable on free CPU hardware.
- Inference time is logged for every call against the 3s budget.
"""
from __future__ import annotations

import time
from typing import Any

from common.config import settings
from common.llm import llm, LLMError
from common.logging_utils import get_logger, log_event
from common.parsing import extract_json, coerce_label

logger = get_logger("triage_agent")

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
INSTRUCTION = "Classify the following customer query and extract urgency and sentiment"

FALLBACK_SYSTEM = (
    "You are a fintech support triage classifier for NovaPay. Classify the "
    "customer query. Return ONLY a JSON object with keys: category (one of "
    "refund/technical/billing), urgency (low/medium/high), sentiment (one of "
    "neutral/frustrated/angry/calm). No markdown, no extra text."
)

_pipeline = None
_pipeline_failed = False


def _get_pipeline():
    """Lazily load the fine-tuned Gemma text-generation pipeline (cached)."""
    global _pipeline, _pipeline_failed
    if _pipeline is not None or _pipeline_failed:
        return _pipeline
    if not settings.use_finetuned_triage:
        # Opt-in only — avoids pulling a 9B model on CPU/local runs.
        _pipeline_failed = True
        return None
    try:
        from transformers import pipeline

        logger.info("Loading fine-tuned triage model '%s'…", settings.triage_hub_repo)
        _pipeline = pipeline(
            "text-generation",
            model=settings.triage_hub_repo,
            tokenizer=settings.triage_hub_repo,
            device_map="auto",
        )
    except Exception as exc:
        logger.warning("Triage model unavailable (%s) — using Claude fallback.", exc)
        _pipeline_failed = True
        _pipeline = None
    return _pipeline


def _normalise(parsed: dict, query: str) -> dict[str, Any]:
    return {
        "category": coerce_label(parsed.get("category"), settings.categories, "technical"),
        "urgency": coerce_label(parsed.get("urgency"), settings.urgencies, "medium"),
        "sentiment": coerce_label(parsed.get("sentiment"), settings.sentiments, "neutral"),
        "summary": (query[:120] + "…") if len(query) > 120 else query,
    }


def _classify_with_gemma(query: str) -> dict | None:
    pipe = _get_pipeline()
    if pipe is None:
        return None
    prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=query)
    out = pipe(prompt, max_new_tokens=128, do_sample=False, return_full_text=False)
    text = out[0]["generated_text"] if out else ""
    parsed = extract_json(text)
    return parsed if isinstance(parsed, dict) else None


def _classify_with_fallback(query: str) -> dict:
    result = llm.complete(
        system=FALLBACK_SYSTEM,
        user=query,
        model=settings.fallback_model,
        max_tokens=200,
        temperature=0.0,
    )
    parsed = extract_json(result.text)
    return parsed if isinstance(parsed, dict) else {}


def triage(query: str) -> dict[str, Any]:
    """Classify ``query``. Returns {category, urgency, sentiment, summary,
    latency_ms, source}."""
    start = time.perf_counter()
    query = (query or "").strip()
    source = "gemma"
    parsed: dict | None = None

    if query:
        try:
            parsed = _classify_with_gemma(query)
        except Exception as exc:
            logger.warning("Gemma inference error: %s", exc)
            parsed = None

        if not parsed:
            source = "fallback"
            log_event("triage_fallbacks.log", query=query, reason="gemma_unavailable_or_unparseable")
            try:
                parsed = _classify_with_fallback(query)
            except LLMError as exc:
                logger.error("Fallback triage failed: %s", exc)
                parsed = {}
                source = "default"

    result = _normalise(parsed or {}, query)
    latency_ms = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(latency_ms, 1)
    result["source"] = source

    if latency_ms > settings.triage_latency_budget_ms:
        logger.warning("Triage exceeded budget: %.0fms > %dms", latency_ms, settings.triage_latency_budget_ms)
    logger.info("Triage [%s] -> %s/%s/%s in %.0fms", source, result["category"],
                result["urgency"], result["sentiment"], latency_ms)
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(triage("My UPI transfer failed but ₹2000 was debited!"), indent=2, ensure_ascii=False))
