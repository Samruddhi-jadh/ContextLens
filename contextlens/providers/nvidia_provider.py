# contextlens/providers/nvidia_provider.py
"""
NVIDIA NIM provider adapter.

NVIDIA's NIM platform exposes an OpenAI-compatible API endpoint,
so we use the openai SDK pointed at NVIDIA's base URL.
This is the cleanest pattern for OpenAI-compatible providers.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from contextlens.config import config
from contextlens.providers.base import BaseProvider, ModelResponse, ProviderError

NVIDIA_PRICING: dict[str, dict[str, float]] = {
    "meta/llama-3.1-70b-instruct":  {"input": 0.35, "output": 0.40},
    "meta/llama-3.1-8b-instruct":   {"input": 0.05, "output": 0.05},
    "mistralai/mixtral-8x7b-instruct-v0.1": {"input": 0.24, "output": 0.24},
    "nvidia/nemotron-4-340b-instruct": {"input": 4.20, "output": 4.20},
}
DEFAULT_NVIDIA_PRICING = {"input": 0.35, "output": 0.40}


class NvidiaProvider(BaseProvider):
    """
    NVIDIA NIM API adapter via OpenAI-compatible SDK.

    NVIDIA NIM hosts open-source models with enterprise-grade
    performance. Uses openai.OpenAI client with a custom base_url.
    """

    def __init__(self, api_key: str | None = None):
        try:
            from openai import OpenAI, APIError, APIConnectionError, RateLimitError
            self._error_types = (APIError, APIConnectionError, RateLimitError)
            OpenAIClient = OpenAI
        except ImportError:
            raise ImportError(
                "openai package not found. Run: pip install openai"
            )

        key = api_key or config.nvidia_api_key
        if not key:
            raise ProviderError(
                "nvidia",
                "NVIDIA_API_KEY not set. Add it to your .env file."
            )

        self._client = OpenAIClient(
            api_key=key,
            base_url=config.nvidia_base_url,
        )
        logger.debug("NvidiaProvider initialized (base_url={}).", config.nvidia_base_url)

    @property
    def provider_name(self) -> str:
        return "nvidia"

    @property
    def default_model(self) -> str:
        return config.nvidia_default_model

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ModelResponse:
        model_name = model or self.default_model
        pricing = NVIDIA_PRICING.get(model_name, DEFAULT_NVIDIA_PRICING)

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
            raise ProviderError("nvidia", str(e), original=e) from e
        except Exception as e:
            raise ProviderError("nvidia", f"Unexpected error: {e}", original=e) from e

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

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