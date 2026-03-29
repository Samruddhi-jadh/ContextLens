# contextlens/providers/__init__.py
"""
Provider registry — maps string names to provider classes.

Senior note: This registry pattern means adding a new provider
is just two steps: write the adapter, add it here.
The router, CLI, and everything else automatically picks it up.
"""

from contextlens.providers.base import BaseProvider, ModelResponse, ProviderError
from contextlens.providers.groq_provider import GroqProvider
from contextlens.providers.gemini_provider import GeminiProvider
from contextlens.providers.nvidia_provider import NvidiaProvider

# Registry: provider_name → provider class
PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "groq":   GroqProvider,
    "gemini": GeminiProvider,
    "nvidia": NvidiaProvider,
}

__all__ = [
    "BaseProvider",
    "ModelResponse",
    "ProviderError",
    "GroqProvider",
    "GeminiProvider",
    "NvidiaProvider",
    "PROVIDER_REGISTRY",
]