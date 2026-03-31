# tests/test_models.py
"""
Tests for provider layer and ModelResponse.

Senior note: We mock the actual SDK clients so tests run
without real API keys — this is the correct pattern for unit tests.
Integration tests (real API calls) live in tests/integration/ and
only run in CI when secrets are available.
"""

from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest

from contextlens.providers.base import ModelResponse, ProviderError, BaseProvider
from contextlens.providers import PROVIDER_REGISTRY


# ---------------------------------------------------------------------------
# ModelResponse tests
# ---------------------------------------------------------------------------

class TestModelResponse:
    def test_cost_per_1k_zero_when_no_tokens(self):
        r = ModelResponse(text="hi", provider="groq", model="llama")
        assert r.cost_per_1k_tokens == 0.0

    def test_cost_per_1k_correct(self):
        r = ModelResponse(
            text="hi", provider="groq", model="llama",
            total_tokens=1000,
            estimated_cost_usd=0.001,
        )
        assert r.cost_per_1k_tokens == pytest.approx(0.001)

    def test_to_dict_has_required_keys(self):
        r = ModelResponse(
            text="Hello", provider="groq", model="llama-3.1",
            input_tokens=100, output_tokens=50, total_tokens=150,
            latency_ms=320.5, estimated_cost_usd=0.000089,
        )
        d = r.to_dict()
        for key in ["text", "provider", "model", "input_tokens",
                    "output_tokens", "latency_ms", "estimated_cost_usd"]:
            assert key in d

    def test_latency_rounded_in_dict(self):
        r = ModelResponse(text="", provider="groq", model="x", latency_ms=123.456789)
        assert r.to_dict()["latency_ms"] == 123.46


# ---------------------------------------------------------------------------
# Provider registry test
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_all_providers_registered(self):
        assert "groq" in PROVIDER_REGISTRY
        assert "gemini" in PROVIDER_REGISTRY
        assert "nvidia" in PROVIDER_REGISTRY

    def test_all_providers_subclass_base(self):
        for name, cls in PROVIDER_REGISTRY.items():
            assert issubclass(cls, BaseProvider), \
                f"{name} must subclass BaseProvider"


# ---------------------------------------------------------------------------
# ProviderError tests
# ---------------------------------------------------------------------------

class TestProviderError:
    def test_message_includes_provider_name(self):
        err = ProviderError("groq", "rate limit exceeded")
        assert "groq" in str(err)
        assert "rate limit exceeded" in str(err)

    def test_original_exception_stored(self):
        original = ValueError("raw sdk error")
        err = ProviderError("gemini", "wrapped", original=original)
        assert err.original is original


# ---------------------------------------------------------------------------
# Groq provider mock test
# ---------------------------------------------------------------------------

class TestGroqProviderMocked:
    @pytest.fixture
    def mock_groq_response(self):
        """Simulate what the Groq SDK returns."""
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150

        choice = MagicMock()
        choice.message.content = "Mocked Groq response"

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    def test_groq_returns_model_response(self, mock_groq_response):
        with patch("contextlens.providers.groq_provider.GroqProvider.__init__",
                   return_value=None):
            from contextlens.providers.groq_provider import GroqProvider
            provider = GroqProvider.__new__(GroqProvider)
            provider._client = MagicMock()
            provider._error_types = (Exception,)
            provider._client.chat.completions.create.return_value = mock_groq_response

            messages = [{"role": "user", "content": "Hello"}]
            result = provider.complete(messages, model="llama-3.1-70b-versatile")

            assert isinstance(result, ModelResponse)
            assert result.text == "Mocked Groq response"
            assert result.provider == "groq"
            assert result.input_tokens == 100
            assert result.output_tokens == 50
            assert result.total_tokens == 150
            assert result.estimated_cost_usd > 0
            assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# Gemini message conversion test
# ---------------------------------------------------------------------------

class TestGeminiMessageConversion:
    def test_system_extracted_correctly(self):
        from contextlens.providers.gemini_provider import GeminiProvider
        messages = [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Explain AI."},
        ]
        system, history = GeminiProvider._convert_messages(messages)
        assert system == "You are an expert."
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["parts"] == ["Explain AI."]

    def test_assistant_becomes_model(self):
        from contextlens.providers.gemini_provider import GeminiProvider
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        _, history = GeminiProvider._convert_messages(messages)
        roles = [m["role"] for m in history]
        assert "assistant" not in roles
        assert "model" in roles
