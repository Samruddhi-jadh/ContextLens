import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def set_test_env(tmp_path, monkeypatch):
    """
    Set safe defaults for all tests running in CI.
    Provides a temp log directory and dummy env vars
    so config.py does not raise on missing keys.
    """
    monkeypatch.setenv("CONTEXTLENS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("CONTEXTLENS_DEFAULT_PROVIDER", "groq")

    # Set dummy keys only if real ones are not present
    # This allows unit tests to run without real API calls
    if not os.getenv("GROQ_API_KEY"):
        monkeypatch.setenv("GROQ_API_KEY", "dummy-key-for-unit-tests")
    if not os.getenv("GEMINI_API_KEY"):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-unit-tests")
    if not os.getenv("NVIDIA_API_KEY"):
        monkeypatch.setenv("NVIDIA_API_KEY", "dummy-key-for-unit-tests")
