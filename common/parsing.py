"""Robust JSON extraction from LLM output.

LLMs sometimes wrap JSON in markdown fences or add a stray sentence. These
helpers recover the JSON payload defensively so a single malformed response
never crashes an agent — it falls back gracefully instead.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Best-effort parse of a JSON object/array embedded in ``text``.

    Tries, in order: direct parse, fenced code block, first balanced
    ``{...}`` or ``[...]`` span. Returns ``None`` if nothing parses.
    """
    if not text:
        return None

    text = text.strip()

    # 1. Straight parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block.
    match = _FENCE_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced brace/bracket span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    return None


def coerce_label(value: str | None, allowed: tuple[str, ...], default: str) -> str:
    """Map a free-text label onto an allowed vocabulary, else ``default``."""
    if not value:
        return default
    v = value.strip().lower()
    if v in allowed:
        return v
    for a in allowed:  # tolerate substrings, e.g. "high urgency"
        if a in v:
            return a
    return default
