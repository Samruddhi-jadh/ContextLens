# contextlens/providers/groq_provider.py
"""
Groq provider adapter.

Groq uses an OpenAI-compatible SDK which makes this the cleanest adapter.
Key Groq advantage: inference speed (often 300+ tokens/sec on Llama models).
We capture latency precisely so the comparison table shows this clearly.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from contextlens.config import config
from contextlens.providers.base import BaseProvider, ModelResponse, ProviderError

# Pricing as of mid-2025 (USD per 1M tokens) — update as Groq changes pricing
GROQ_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile":  {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":     {"input": 0.05, "output": 0.08},
    "llama3-70b-8192":          {"input": 0.59, "output": 0.79},
    "mixtral-8x7b-32768":       {"input": 0.24, "output": 0.24},
    "gemma2-9b-it":             {"input": 0.20, "output": 0.20},
}
DEFAULT_GROQ_PRICING = {"input": 0.59, "output": 0.79}  # fallback


class GroqProvider(BaseProvider):
    """
    Groq API adapter.

    Uses the official groq Python SDK (openai-compatible interface).
    Handles auth, error wrapping, token extraction, and cost estimation.
    """

    def __init__(self, api_key: str | None = None):
        try:
            from groq import Groq, APIError, APIConnectionError, RateLimitError
            self._error_types = (APIError, APIConnectionError, RateLimitError)
        except ImportError:
            raise ImportError(
                "groq package not found. Run: pip install groq"
            )

        key = api_key or config.groq_api_key
        if not key:
            raise ProviderError(
                "groq",
                "GROQ_API_KEY not set. Add it to your .env file."
            )

        from groq import Groq
        self._client = Groq(api_key=key)
        logger.debug("GroqProvider initialized.")

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def default_model(self) -> str:
        return config.groq_default_model

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ModelResponse:
        model_name = model or self.default_model
        pricing = GROQ_PRICING.get(model_name, DEFAULT_GROQ_PRICING)

        try:
            start = time.perf_counter()
            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            latency_ms = (time.perf_counter() - start) * 1000

        except self._error_types as e:
            raise ProviderError("groq", str(e), original=e) from e
        except Exception as e:
            raise ProviderError("groq", f"Unexpected error: {e}", original=e) from e

        # Extract usage (Groq always returns this)
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        # Cost estimation
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        return ModelResponse(
            text=response.choices[0].message.content or "",
            provider=self.provider_name,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            raw_response=response,
        )