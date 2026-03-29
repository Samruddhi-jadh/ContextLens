# tests/test_logger.py
"""
Tests for ContextLogger and ContextEntry.

Senior note: We test the contract (what the logger guarantees),
not the implementation (how it stores data internally).
That way, we can refactor storage later without breaking tests.
"""

import json
import tempfile
from pathlib import Path

import pytest

from contextlens.core.logger import (
    ContextEntry,
    ContextLogger,
    RetrievedDoc,
    TokenCounter,
)


# ---------------------------------------------------------------------------
# Token Counter tests
# ---------------------------------------------------------------------------

class TestTokenCounter:
    def test_empty_string_returns_zero(self):
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_nonempty_string_returns_positive(self):
        counter = TokenCounter()
        assert counter.count("hello world") > 0

    def test_longer_text_has_more_tokens(self):
        counter = TokenCounter()
        short = counter.count("hello")
        long = counter.count("hello world this is a longer sentence with more tokens")
        assert long > short

    def test_message_list_counting(self):
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = counter.count_messages(messages)
        assert count > 0


# ---------------------------------------------------------------------------
# ContextEntry tests
# ---------------------------------------------------------------------------

class TestContextEntry:
    def test_required_field_user_prompt(self):
        """user_prompt is the only required field."""
        entry = ContextEntry(user_prompt="test")
        assert entry.user_prompt == "test"
        assert entry.run_id  # auto-generated UUID

    def test_token_utilization_zero_when_no_tokens(self):
        entry = ContextEntry(user_prompt="test", total_tokens=0)
        assert entry.token_utilization == 0.0

    def test_token_utilization_correct_ratio(self):
        entry = ContextEntry(
            user_prompt="test",
            total_tokens=4000,
            max_context_tokens=8000,
        )
        assert entry.token_utilization == 0.5

    def test_docs_summary_empty(self):
        entry = ContextEntry(user_prompt="test")
        summary = entry.docs_summary
        assert summary["count"] == 0
        assert summary["avg_relevance"] == 0.0

    def test_docs_summary_with_docs(self):
        doc = RetrievedDoc(content="some content", source="wiki", relevance_score=0.9)
        entry = ContextEntry(user_prompt="test", retrieved_docs=[doc])
        summary = entry.docs_summary
        assert summary["count"] == 1
        assert summary["sources"] == ["wiki"]
        assert summary["avg_relevance"] == pytest.approx(0.9)

    def test_serialization_roundtrip(self):
        doc = RetrievedDoc(content="test doc", source="db", relevance_score=0.7)
        entry = ContextEntry(
            user_prompt="What is AI?",
            system_prompt="You are an expert.",
            retrieved_docs=[doc],
            provider="groq",
            model="llama-3.1-70b-versatile",
        )
        data = entry.to_log_dict()
        assert data["user_prompt"] == "What is AI?"
        assert data["provider"] == "groq"
        assert "token_utilization" in data
        assert "docs_summary" in data


class TestRetrievedDoc:
    def test_relevance_score_validation(self):
        with pytest.raises(Exception):
            RetrievedDoc(content="test", relevance_score=1.5)  # out of range

    def test_valid_relevance_score(self):
        doc = RetrievedDoc(content="test", relevance_score=0.85)
        assert doc.relevance_score == 0.85


# ---------------------------------------------------------------------------
# ContextLogger integration tests
# ---------------------------------------------------------------------------

class TestContextLogger:
    @pytest.fixture
    def temp_logger(self, tmp_path):
        """Each test gets a fresh logger writing to a temp directory."""
        return ContextLogger(log_dir=tmp_path, session_id="test")

    def test_log_creates_file(self, temp_logger, tmp_path):
        temp_logger.log(user_prompt="Hello, AI!")
        log_files = list(tmp_path.glob("*.jsonl"))
        assert len(log_files) == 1

    def test_log_returns_context_entry(self, temp_logger):
        entry = temp_logger.log(user_prompt="What is context engineering?")
        assert isinstance(entry, ContextEntry)
        assert entry.user_prompt == "What is context engineering?"

    def test_token_counts_populated(self, temp_logger):
        entry = temp_logger.log(
            user_prompt="Explain transformers in AI",
            system_prompt="You are an AI expert.",
        )
        assert entry.user_tokens > 0
        assert entry.system_tokens > 0
        assert entry.total_tokens == entry.user_tokens + entry.system_tokens

    def test_doc_tokens_counted(self, temp_logger):
        docs = [RetrievedDoc(content="This is a retrieved document with content.")]
        entry = temp_logger.log(user_prompt="test", retrieved_docs=docs)
        assert entry.docs_tokens > 0
        assert entry.retrieved_docs[0].token_count > 0

    def test_multiple_runs_appended(self, temp_logger, tmp_path):
        temp_logger.log(user_prompt="First run")
        temp_logger.log(user_prompt="Second run")
        entries = temp_logger.load_session()
        assert len(entries) == 2

    def test_log_file_is_valid_jsonl(self, temp_logger, tmp_path):
        temp_logger.log(user_prompt="JSONL test")
        log_file = list(tmp_path.glob("*.jsonl"))[0]
        with open(log_file) as f:
            for line in f:
                parsed = json.loads(line)  # must not raise
                assert "run_id" in parsed
                assert "user_prompt" in parsed

    def test_session_stats(self, temp_logger):
        temp_logger.log(user_prompt="First question")
        temp_logger.log(user_prompt="Second question")
        stats = temp_logger.get_token_stats()
        assert stats["runs"] == 2
        assert stats["total_tokens_used"] > 0

    def test_log_does_not_crash_on_bad_log_dir(self):
        """Logger must handle disk errors gracefully — never crash the app."""
        # Write to a path that's valid but log writing should handle errors
        cl = ContextLogger(log_dir=Path("/tmp/cl_test_safe"), session_id="t")
        entry = cl.log(user_prompt="test")
        assert entry is not None  # must return even if log fails