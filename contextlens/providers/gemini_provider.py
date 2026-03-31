# contextlens/providers/gemini_provider.py
"""
Google Gemini provider adapter.

Gemini uses its own SDK (google-generativeai), not OpenAI-compatible.
Key challenge: Gemini has a different message format — it uses
'parts' and 'user'/'model' roles instead of 'user'/'assistant'.
We handle that conversion here so the router never sees it.
"""

from __future__ import annotations

import time
from typing import Any
from google import genai

from loguru import logger

from contextlens.config import config
from contextlens.providers.base import BaseProvider, ModelResponse, ProviderError

# Gemini pricing (USD per 1M tokens, mid-2025 estimates)
GEMINI_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash":         {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b":      {"input": 0.0375, "output": 0.15},
    "gemini-1.5-pro":           {"input": 3.50,  "output": 10.50},
    "gemini-2.0-flash-exp":     {"input": 0.075, "output": 0.30},
}
DEFAULT_GEMINI_PRICING = {"input": 0.075, "output": 0.30}


class GeminiProvider(BaseProvider):
    """
    Google Gemini API adapter.

    Converts OpenAI-format messages to Gemini format internally.
    Extracts system prompt separately (Gemini requires this).
    """

    def __init__(self, api_key: str | None = None):
    try:
        import google.generativeai as genai
        self._genai = genai
        self._sdk_version = "legacy"
    except ImportError:
        raise ImportError(
            "google-generativeai not found. Run: pip install google-generativeai"
        )

    key = api_key or config.gemini_api_key
    if not key:
        raise ProviderError(
            "gemini",
            "GEMINI_API_KEY not set. Add it to your .env file."
        )

    self._genai.configure(api_key=key)
    logger.debug("GeminiProvider initialized.")

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return config.gemini_default_model

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str, list[dict]]:
        """
        Convert OpenAI-format messages to Gemini format.

        OpenAI:  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        Gemini:  system_instruction="...", history=[{"role":"user","parts":["..."]}]

        Returns: (system_instruction, gemini_history)
        """
        system_instruction = ""
        history = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Gemini takes system prompt separately
                system_instruction = content
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                history.append({"role": "model", "parts": [content]})

        return system_instruction, history

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ModelResponse:
        model_name = model or self.default_model
        pricing = GEMINI_PRICING.get(model_name, DEFAULT_GEMINI_PRICING)

        system_instruction, history = self._convert_messages(messages)

        # Pull the last user message as the actual prompt
        # (Gemini's send_message takes the current turn separately)
        if not history or history[-1]["role"] != "user":
            raise ProviderError("gemini", "Last message must be from 'user'.")

        current_prompt_parts = history[-1]["parts"]
        prior_history = history[:-1]

        try:
            generation_config = self._genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            model_instance = self._genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction or None,
                generation_config=generation_config,
            )

            chat = model_instance.start_chat(history=prior_history)

            start = time.perf_counter()
            response = chat.send_message(current_prompt_parts)
            latency_ms = (time.perf_counter() - start) * 1000

        except Exception as e:
            raise ProviderError("gemini", str(e), original=e) from e

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (input_tokens + output_tokens)

        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        return ModelResponse(
            text=response.text or "",
            provider=self.provider_name,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            raw_response=response,
        )
