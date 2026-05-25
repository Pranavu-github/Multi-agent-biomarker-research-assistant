"""
utils/llm_client.py — Central LLM caller.

Priority:
  1. Anthropic Claude (primary)
  2. Groq (fallback if Anthropic key missing or quota exceeded)

Groq free-tier note:
  - llama-3.3-70b-versatile: ~6,000 TPM / 30 RPM on free tier
  - Rate-limit errors (HTTP 429) are retried with exponential backoff (up to 3 attempts)
    so callers never need to handle rate limiting themselves.
  - If all retries fail, RuntimeError is raised.

Usage:
    from utils.llm_client import call_llm
    response = call_llm("Your prompt here")
"""

from __future__ import annotations

import os
import time
from loguru import logger


# Groq model tiers:
# - FAST: llama-3.1-8b-instant   → used for extraction (many calls, speed matters)
# - QUALITY: llama-3.3-70b-versatile → used for planner + scoring (fewer calls, quality matters)
_GROQ_MODEL_FAST    = "llama-3.1-8b-instant"
_GROQ_MODEL_QUALITY = "llama-3.3-70b-versatile"
_GROQ_MODEL         = _GROQ_MODEL_FAST   # default for most calls

# Retry settings for Groq 429 / rate-limit responses
_GROQ_MAX_RETRIES   = 3
_GROQ_RETRY_BASE_S  = 2   # seconds — doubles each retry: 2s, 4s, 8s (reduced from 4s)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a Groq 429 / rate-limit error."""
    msg = str(exc).lower()
    return any(k in msg for k in ("rate_limit", "rate limit", "429", "too many requests"))


def call_llm(
    prompt:      str,
    model:       str   = "claude-sonnet-4-6",
    max_tokens:  int   = 1024,
    system:      str   = "",
    temperature: float = 1.0,
    use_fast:    bool  = True,   # True = 8b-instant (fast), False = 70b (quality)
) -> str:
    """
    Call the configured LLM and return the response text.

    Args:
        prompt:      User/human turn content.
        model:       Model name (Anthropic or Groq).
        max_tokens:  Maximum tokens to generate.
        system:      Optional system prompt.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = default).
                     Use low values (0.05–0.10) for structured JSON tasks like
                     query generation where consistency matters more than creativity.

    Returns:
        Stripped response string.

    Raises:
        RuntimeError: If no API key is available or all retries exhausted.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    groq_key      = os.environ.get("GROQ_API_KEY", "")

    # ── Try Anthropic first ──────────────────────────────────────────────────
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)

            messages_kwargs: dict = {
                "model":       model,
                "max_tokens":  max_tokens,
                "temperature": temperature,
                "messages":    [{"role": "user", "content": prompt}],
            }
            if system:
                messages_kwargs["system"] = system

            response = client.messages.create(**messages_kwargs)
            return response.content[0].text.strip()

        except Exception as exc:
            logger.warning(f"[llm_client] Anthropic call failed: {exc} — trying Groq fallback")

    # ── Groq fallback — with exponential-backoff retry on rate-limit errors ──
    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        groq_model = _GROQ_MODEL_FAST if use_fast else _GROQ_MODEL_QUALITY
        logger.info(f"[llm_client] Using Groq model: {groq_model}")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc: Exception | None = None

        for attempt in range(_GROQ_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=groq_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()

            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc) and attempt < _GROQ_MAX_RETRIES - 1:
                    wait = _GROQ_RETRY_BASE_S * (2 ** attempt)   # 4s → 8s → 16s
                    logger.warning(
                        f"[llm_client] Groq rate-limit hit (attempt {attempt + 1}/"
                        f"{_GROQ_MAX_RETRIES}) — retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    # Non-rate-limit error OR final retry — give up immediately
                    logger.error(f"[llm_client] Groq call failed (attempt {attempt + 1}): {exc}")
                    break

        if last_exc is not None:
            raise RuntimeError(f"[llm_client] Groq: all {_GROQ_MAX_RETRIES} attempts failed. Last error: {last_exc}")

    raise RuntimeError(
        "[llm_client] No working LLM API key found. "
        "Set ANTHROPIC_API_KEY (and optionally GROQ_API_KEY) in your .env file."
    )
