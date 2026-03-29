# contextlens/config.py
"""
Centralized configuration using Pydantic Settings.
All config is loaded from environment variables / .env file.
This is the ONLY place in the codebase where env vars are read.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ContextLensConfig(BaseSettings):
    """
    Production-grade config using Pydantic v2 Settings.
    Validates types, provides defaults, fails fast on missing required values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars
    )

    # --- API Keys ---
    groq_api_key: str = Field(default="", description="Groq API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    nvidia_api_key: str = Field(default="", description="NVIDIA NIM API key")

    # --- Default Models ---
    groq_default_model: str = "llama-3.3-70b-versatile"
    gemini_default_model: str = "gemini-2.5-flash"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_default_model: str = "meta/llama-3.1-70b-instruct"

    # --- ContextLens Behaviour ---
    log_dir: Path = Field(default=Path("./logs"), alias="CONTEXTLENS_LOG_DIR")
    max_context_tokens: int = Field(default=8000, alias="CONTEXTLENS_MAX_CONTEXT_TOKENS")
    default_provider: str = Field(default="groq", alias="CONTEXTLENS_DEFAULT_PROVIDER")
    enable_optimization: bool = Field(default=True, alias="CONTEXTLENS_ENABLE_OPTIMIZATION")

    def validate_provider(self, provider: str) -> None:
        """Raise early if a provider is requested without its API key."""
        key_map = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "nvidia": self.nvidia_api_key,
        }
        if provider not in key_map:
            raise ValueError(f"Unknown provider '{provider}'. Choose: {list(key_map)}")
        if not key_map[provider]:
            raise EnvironmentError(
                f"Provider '{provider}' requires an API key. "
                f"Set {provider.upper()}_API_KEY in your .env file."
            )

    def ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere instead of re-reading env vars
config = ContextLensConfig()
config.ensure_log_dir()