"""Provider-agnostic LLM client.

Every LLM call in the system goes through :meth:`LLMClient.complete`. The client
hides the vendor behind one interface (Adapter pattern), so switching from Groq
to Anthropic/Gemini/Ollama is a config change, not a code change. In one place
it provides:

- lazy, cached client construction (importing a module never needs a key);
- exponential backoff with jitter on transient/rate-limit errors;
- uniform system/user prompt handling across vendors;
- token-usage accounting returned with the text (real $/ticket downstream);
- a typed :class:`LLMResult` so callers never touch a raw SDK object.

Supported providers (set ``LLM_PROVIDER`` or ``settings.llm_provider``):
- ``groq``      — free tier, OpenAI-compatible (default).
- ``anthropic`` — paid; Claude.
- ``gemini``    — free tier; Google Generative AI.
- ``ollama``    — fully local; no key, uses an OpenAI-compatible local endpoint.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from common.config import settings
from common.logging_utils import get_logger

logger = get_logger("llm")


@dataclass
class LLMResult:
    """Normalised result of a single completion."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return settings.cost_usd(self.model, self.input_tokens, self.output_tokens)


class LLMError(RuntimeError):
    """Raised when a completion fails after exhausting retries."""


class LLMClient:
    """Singleton-friendly wrapper that adapts several vendor SDKs."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or settings.llm_provider
        self._client = None  # built lazily on first use

    # -- client construction ---------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "groq":
            if not settings.groq_api_key:
                raise LLMError("GROQ_API_KEY is not set. Get a free key at console.groq.com and add it to .env.")
            try:
                from groq import Groq
            except ImportError as exc:
                raise LLMError("The 'groq' package is not installed. Run `pip install groq`.") from exc
            self._client = Groq(api_key=settings.groq_api_key)

        elif self.provider == "anthropic":
            if not settings.anthropic_api_key:
                raise LLMError("ANTHROPIC_API_KEY is not set. Add it to .env.")
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError("The 'anthropic' package is not installed. Run `pip install anthropic`.") from exc
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.request_timeout_s)

        elif self.provider == "gemini":
            if not settings.gemini_api_key:
                raise LLMError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it to .env.")
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise LLMError("Run `pip install google-generativeai`.") from exc
            genai.configure(api_key=settings.gemini_api_key)
            self._client = genai

        elif self.provider == "ollama":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("Run `pip install openai` (Ollama uses an OpenAI-compatible API).") from exc
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self._client = OpenAI(base_url=base_url, api_key="ollama")

        else:
            raise LLMError(f"Unknown LLM provider: {self.provider!r}")

        return self._client

    # -- per-provider raw calls ------------------------------------------
    def _call(self, client, model, system, user, max_tokens, temperature):
        """Return (text, input_tokens, output_tokens, stop_reason)."""
        if self.provider in ("groq", "ollama"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            )
            choice = resp.choices[0]
            usage = resp.usage
            return (
                choice.message.content or "",
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
                choice.finish_reason,
            )

        if self.provider == "anthropic":
            kwargs = {
                "model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "user", "content": user}],
            }
            if system:
                kwargs["system"] = system
            resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            return text, resp.usage.input_tokens, resp.usage.output_tokens, getattr(resp, "stop_reason", None)

        if self.provider == "gemini":
            gm = client.GenerativeModel(model_name=model, system_instruction=system or None)
            resp = gm.generate_content(
                user,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
            )
            um = getattr(resp, "usage_metadata", None)
            in_tok = getattr(um, "prompt_token_count", 0) if um else 0
            out_tok = getattr(um, "candidates_token_count", 0) if um else 0
            return resp.text, in_tok, out_tok, None

        raise LLMError(f"Unknown LLM provider: {self.provider!r}")

    def _retryable_errors(self):
        """Vendor exception classes worth retrying. Falls back to Exception."""
        try:
            if self.provider == "groq":
                from groq import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
                return (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
            if self.provider == "anthropic":
                import anthropic
                return (anthropic.RateLimitError, anthropic.APITimeoutError,
                        anthropic.APIConnectionError, anthropic.InternalServerError)
        except Exception:
            pass
        return (Exception,)

    # -- public -----------------------------------------------------------
    def complete(
        self,
        *,
        user: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> LLMResult:
        """Run a single-turn completion with retries and token accounting."""
        model = model or settings.primary_model
        max_tokens = max_tokens or settings.api_max_tokens
        client = self._get_client()
        retryable = self._retryable_errors()

        last_exc: Exception | None = None
        for attempt in range(settings.api_max_retries):
            start = time.perf_counter()
            try:
                text, in_tok, out_tok, stop = self._call(
                    client, model, system, user, max_tokens, temperature
                )
                return LLMResult(
                    text=(text or "").strip(), model=model,
                    input_tokens=in_tok, output_tokens=out_tok,
                    latency_ms=(time.perf_counter() - start) * 1000, stop_reason=stop,
                )
            except retryable as exc:  # type: ignore[misc]
                last_exc = exc
                delay = settings.api_base_delay_s * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("[%s] API call failed (attempt %d/%d): %s — retry in %.1fs",
                               self.provider, attempt + 1, settings.api_max_retries, exc, delay)
                time.sleep(delay)

        raise LLMError(f"Completion failed after retries: {last_exc}") from last_exc


# Module-level singleton — import this everywhere.
llm = LLMClient()
