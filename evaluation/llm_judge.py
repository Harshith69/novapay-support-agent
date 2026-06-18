"""LLM-as-judge.

Scores a single agent response on faithfulness, helpfulness and tone (1–5
each) against the knowledge-base context it used, returning a structured
verdict. All judgements are appended to ``evaluation/judge_log.jsonl``.
"""
from __future__ import annotations

from typing import Any

from common.config import EVALUATION_DIR, settings
from common.llm import llm, LLMError
from common.logging_utils import get_logger, append_jsonl, utc_now_iso
from common.parsing import extract_json

logger = get_logger("llm_judge")

SYSTEM_PROMPT = (
    "You are an expert evaluator for a fintech customer support AI. Score the "
    "given agent response on three dimensions. Return only a JSON object with no "
    "markdown."
)


def judge_response(query: str, response: str, context_chunks: list[str]) -> dict[str, Any]:
    context = "\n".join(context_chunks) if context_chunks else "(none)"
    user_prompt = (
        f"Query: {query}\nAgent Response: {response}\nKnowledge Base Context Used: {context}\n\n"
        "Score on: (1) faithfulness: does the response contain only information present "
        "in the context? Score 1-5. (2) helpfulness: does the response actually address "
        "the customer's problem? Score 1-5. (3) tone: is the tone empathetic and "
        "professional? Score 1-5. Also add a 'verdict' field: pass if all scores >= 3, "
        "fail otherwise. Add a 'notes' field with one sentence of feedback."
    )
    default = {"faithfulness": 0, "helpfulness": 0, "tone": 0, "verdict": "fail", "notes": "judge_error"}

    try:
        result = llm.complete(
            system=SYSTEM_PROMPT, user=user_prompt,
            model=settings.primary_model, max_tokens=300, temperature=0.0,
        )
    except LLMError as exc:
        logger.error("Judge failed: %s", exc)
        return default

    parsed = extract_json(result.text)
    if not isinstance(parsed, dict):
        parsed = default

    append_jsonl(EVALUATION_DIR / "judge_log.jsonl", {
        "timestamp": utc_now_iso(), "query": query, "response": response, **parsed,
    })
    return parsed


if __name__ == "__main__":
    import json

    demo = judge_response(
        "My UPI failed but money debited",
        "I understand your frustration. UPI failures are auto-reversed within 3 business days.",
        ["UPI auto-reversal happens within 3 business days."],
    )
    print(json.dumps(demo, indent=2, ensure_ascii=False))
