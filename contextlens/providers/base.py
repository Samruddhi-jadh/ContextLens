# contextlens/providers/base.py
"""
Abstract base provider.

Senior design principle: program to an interface, not an implementation.
Every provider MUST implement these methods. This is what lets the router
swap providers without changing any downstream code — the evaluator,
optimizer, and CLI all just work with ModelResponse regardless of source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """
    Normalized response from any AI provider.

    Design decision: We flatten provider-specific response shapes into
    this one structure. This is the ONLY type the rest of ContextLens
    ever sees — no provider-specific objects leak out of this layer.
    """
    # Core output
    text: str                          # The actual response content
    provider: str                      # "groq" | "gemini" | "nvidia"
    model: str                         # exact model string used

    # Usage metrics (populated from provider response headers/fields)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Performance
    latency_ms: float = 0.0           # wall-clock time for the API call
    
    # Cost estimation (USD) — calculated from known pricing
    estimated_cost_usd: float = 0.0

    # Run linkage — ties back to the ContextEntry that produced this
    run_id: str = ""

    # Raw provider response (for debugging — never shown to end user)
    raw_response: Any = field(default=None, repr=False)

    @property
    def cost_per_1k_tokens(self) -> float:
        """Cost efficiency metric for provider comparison."""
        if self.total_tokens == 0:
            return 0.0
        return (self.estimated_cost_usd / self.total_tokens) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "run_id": self.run_id,
        }


class BaseProvider(ABC):
    """
    Abstract base class for all AI providers.

    Any new provider (Anthropic, Cohere, Mistral...) just needs to
    implement these two methods and it automatically works with
    the entire ContextLens pipeline.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique string identifier. e.g. 'groq', 'gemini', 'nvidia'"""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model string for this provider."""
        ...

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Send messages to the provider, return a normalized ModelResponse.

        Args:
            messages:    OpenAI-format message list:
                         [{"role": "system", "content": "..."}, ...]
            model:       Override the default model.
            max_tokens:  Max response tokens.
            temperature: Sampling temperature (0.0 = deterministic).
            **kwargs:    Provider-specific extras.

        Returns:
            ModelResponse: Normalized, provider-agnostic response.

        Raises:
            ProviderError: Wraps all provider-specific exceptions.
        """
        ...

    def health_check(self) -> bool:
        """
        Quick check that the provider is reachable with current credentials.
        Used by the CLI 'contextlens check' command.
        """
        try:
            response = self.complete(
                messages=[{"role": "user", "content": "Reply with OK"}],
                max_tokens=5,
                temperature=0.0,
            )
            return bool(response.text)
        except Exception:
            return False


class ProviderError(Exception):
    """
    Unified error type for all provider failures.

    Senior note: We never let provider-specific exceptions (groq.APIError,
    google.api_core.exceptions.*) leak out of the provider layer.
    Everything becomes a ProviderError so callers have one thing to catch.
    """
    def __init__(self, provider: str, message: str, original: Exception | None = None):
        self.provider = provider
        self.original = original
        super().__init__(f"[{provider}] {message}")