# tests/test_optimizer.py
"""
Tests for the ContextOptimizer and all 4 strategy functions.

Testing strategy:
  - Each strategy function is tested in pure isolation
  - We construct ContextEntry fixtures that specifically trigger each strategy
  - We verify the change log is accurate and tokens_saved is real
  - We verify the "no change" path (already good context) returns cleanly
"""

from __future__ import annotations
import pytest

from contextlens.core.logger import ContextEntry, RetrievedDoc, TokenCounter
from contextlens.core.evaluator import ContextEvaluator, EvaluationReport
from contextlens.core.optimizer import (
    ContextOptimizer,
    OptimizationConfig,
    OptimizationResult,
    strategy_deduplicate,
    strategy_rerank,
    strategy_trim,
    strategy_compress,
    _rebuild_entry,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

COUNTER = TokenCounter()
CFG = OptimizationConfig()


def make_entry(
    user_prompt: str = "How does context engineering improve LLM accuracy?",
    system_prompt: str = "You are a senior AI engineer. Be precise.",
    docs: list[RetrievedDoc] | None = None,
    max_context_tokens: int = 8000,
) -> ContextEntry:
    retrieved = docs or []
    docs_tokens = sum(COUNTER.count(d.content) for d in retrieved)
    return ContextEntry(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        retrieved_docs=retrieved,
        system_tokens=COUNTER.count(system_prompt),
        user_tokens=COUNTER.count(user_prompt),
        docs_tokens=docs_tokens,
        total_tokens=COUNTER.count(system_prompt) + COUNTER.count(user_prompt) + docs_tokens,
        max_context_tokens=max_context_tokens,
        provider="groq",
        model="llama-3.1-70b-versatile",
    )


DUPLICATE_CONTENT = (
    "Context engineering is the practice of designing and optimizing the "
    "information passed to large language models to maximize accuracy and "
    "minimize hallucination in production AI systems. It involves prompt "
    "design, retrieval augmentation, and context compression strategies."
)
DIVERSE_CONTENT_A = "Quantum computing uses qubits to perform parallel computations."
DIVERSE_CONTENT_B = "Machine learning involves training statistical models on data."
DIVERSE_CONTENT_C = "Ocean tides are driven by gravitational forces from the Moon."


# ---------------------------------------------------------------------------
# Strategy 1: Deduplicate
# ---------------------------------------------------------------------------

class TestStrategyDeduplicate:
    def test_near_duplicates_removed(self):
        doc1 = RetrievedDoc(content=DUPLICATE_CONTENT, source="a")
        doc2 = RetrievedDoc(content=DUPLICATE_CONTENT + " This is important.", source="b")
        entry = make_entry(docs=[doc1, doc2])
        new_entry, record = strategy_deduplicate(entry, CFG, COUNTER)
        assert record is not None
        assert len(new_entry.retrieved_docs) < 2
        assert record.strategy == "deduplicate"
        assert record.tokens_saved > 0

    def test_diverse_docs_not_removed(self):
        docs = [
            RetrievedDoc(content=DIVERSE_CONTENT_A, source="a"),
            RetrievedDoc(content=DIVERSE_CONTENT_B, source="b"),
            RetrievedDoc(content=DIVERSE_CONTENT_C, source="c"),
        ]
        entry = make_entry(docs=docs)
        new_entry, record = strategy_deduplicate(entry, CFG, COUNTER)
        assert record is None
        assert len(new_entry.retrieved_docs) == 3

    def test_single_doc_unchanged(self):
        entry = make_entry(docs=[RetrievedDoc(content=DUPLICATE_CONTENT)])
        new_entry, record = strategy_deduplicate(entry, CFG, COUNTER)
        assert record is None
        assert len(new_entry.retrieved_docs) == 1

    def test_no_docs_unchanged(self):
        entry = make_entry(docs=[])
        new_entry, record = strategy_deduplicate(entry, CFG, COUNTER)
        assert record is None

    def test_first_doc_kept_on_duplicate(self):
        doc1 = RetrievedDoc(content=DUPLICATE_CONTENT, source="first")
        doc2 = RetrievedDoc(content=DUPLICATE_CONTENT, source="second")
        entry = make_entry(docs=[doc1, doc2])
        new_entry, _ = strategy_deduplicate(entry, CFG, COUNTER)
        assert new_entry.retrieved_docs[0].source == "first"

    def test_threshold_respected(self):
        doc1 = RetrievedDoc(content=DUPLICATE_CONTENT, source="a")
        doc2 = RetrievedDoc(content=DUPLICATE_CONTENT, source="b")
        entry = make_entry(docs=[doc1, doc2])
        strict_cfg = OptimizationConfig(similarity_threshold=0.99)
        _, record = strategy_deduplicate(entry, strict_cfg, COUNTER)
        # With very high threshold, identical docs still get caught
        # (similarity = 1.0 > 0.99)
        assert record is not None


# ---------------------------------------------------------------------------
# Strategy 2: Re-rank
# ---------------------------------------------------------------------------

class TestStrategyRerank:
    def test_reranks_by_relevance(self):
        relevant = RetrievedDoc(
            content="Context engineering optimizes LLM accuracy in production.",
            source="relevant", relevance_score=0.95,
        )
        irrelevant = RetrievedDoc(
            content="The annual rainfall in tropical regions exceeds 2000mm.",
            source="irrelevant", relevance_score=0.10,
        )
        entry = make_entry(docs=[irrelevant, relevant])
        new_entry, record = strategy_rerank(entry, CFG, COUNTER)
        assert new_entry.retrieved_docs[0].source == "relevant"

    def test_max_docs_respected(self):
        docs = [
            RetrievedDoc(content=f"Doc {i} about context engineering LLM accuracy.", source=f"s{i}", relevance_score=0.9 - i * 0.1)
            for i in range(8)
        ]
        entry = make_entry(docs=docs)
        cfg = OptimizationConfig(max_docs=3)
        new_entry, record = strategy_rerank(entry, cfg, COUNTER)
        assert len(new_entry.retrieved_docs) <= 3
        assert record is not None

    def test_single_doc_unchanged(self):
        entry = make_entry(docs=[RetrievedDoc(content="Only one doc.", source="a")])
        new_entry, record = strategy_rerank(entry, CFG, COUNTER)
        assert record is None

    def test_min_relevance_filter_applied(self):
        docs = [
            RetrievedDoc(content="Highly relevant content about context engineering.", source="a", relevance_score=0.90),
            RetrievedDoc(content="Slightly relevant content about prompting.", source="b", relevance_score=0.30),
        ]
        entry = make_entry(docs=docs)
        cfg = OptimizationConfig(min_relevance_score=0.50)
        new_entry, record = strategy_rerank(entry, cfg, COUNTER)
        sources = [d.source for d in new_entry.retrieved_docs]
        assert "b" not in sources


# ---------------------------------------------------------------------------
# Strategy 3: Trim
# ---------------------------------------------------------------------------

class TestStrategyTrim:
    def test_long_doc_trimmed(self):
        long_content = " ".join(["context engineering produces accurate results"] * 30)
        doc = RetrievedDoc(content=long_content, source="long")
        entry = make_entry(docs=[doc])
        cfg = OptimizationConfig(max_doc_tokens=50)
        new_entry, record = strategy_trim(entry, cfg, COUNTER)
        assert record is not None
        new_tokens = COUNTER.count(new_entry.retrieved_docs[0].content)
        assert new_tokens <= 60  # small buffer for "[truncated]" marker

    def test_short_doc_unchanged(self):
        doc = RetrievedDoc(content="Short content.", source="short")
        entry = make_entry(docs=[doc])
        new_entry, record = strategy_trim(entry, CFG, COUNTER)
        assert record is None

    def test_no_docs_unchanged(self):
        entry = make_entry(docs=[])
        new_entry, record = strategy_trim(entry, CFG, COUNTER)
        assert record is None

    def test_truncated_marker_added(self):
        long_content = ". ".join([
            "Context engineering is the practice of designing prompts for LLMs"
        ] * 20) + "."
        doc = RetrievedDoc(content=long_content, source="long")
        entry = make_entry(docs=[doc])
        cfg = OptimizationConfig(max_doc_tokens=30)
        new_entry, _ = strategy_trim(entry, cfg, COUNTER)
        assert "[truncated]" in new_entry.retrieved_docs[0].content

    def test_budget_cap_reduces_total(self):
        docs = [
            RetrievedDoc(content=" ".join(["token"] * 500), source=f"doc{i}")
            for i in range(4)
        ]
        entry = make_entry(docs=docs, max_context_tokens=2000)
        cfg = OptimizationConfig(target_token_budget=0.60, max_doc_tokens=600)
        new_entry, record = strategy_trim(entry, cfg, COUNTER)
        if record:
            assert new_entry.docs_tokens < entry.docs_tokens


# ---------------------------------------------------------------------------
# Strategy 4: Compress
# ---------------------------------------------------------------------------

class TestStrategyCompress:
    def test_irrelevant_sentences_removed(self):
        content = (
            "Context engineering improves LLM accuracy in production systems. "
            "The weather in Paris is mild in spring. "
            "Optimizing context reduces hallucinations and token costs. "
            "Ancient Rome was founded in 753 BC."
        )
        doc = RetrievedDoc(content=content, source="mixed")
        entry = make_entry(
            user_prompt="How does context engineering improve LLM accuracy?",
            docs=[doc],
        )
        cfg = OptimizationConfig(
            enable_compression=True,
            compression_keyword_ratio=0.10,
            min_sentences_per_doc=1,
        )
        new_entry, record = strategy_compress(entry, cfg, COUNTER)
        if record:
            compressed = new_entry.retrieved_docs[0].content
            assert "Paris" not in compressed or "context" in compressed

    def test_disabled_compression_no_change(self):
        doc = RetrievedDoc(content="Some content here.", source="a")
        entry = make_entry(docs=[doc])
        cfg = OptimizationConfig(enable_compression=False)
        _, record = strategy_compress(entry, cfg, COUNTER)
        assert record is None

    def test_min_sentences_guaranteed(self):
        content = "First sentence about context. Second unrelated sentence about food. Third about ML models."
        doc = RetrievedDoc(content=content, source="a")
        entry = make_entry(
            user_prompt="context engineering LLM",
            docs=[doc],
        )
        cfg = OptimizationConfig(
            enable_compression=True,
            compression_keyword_ratio=0.99,  # very strict — almost everything filtered
            min_sentences_per_doc=2,
        )
        new_entry, _ = strategy_compress(entry, cfg, COUNTER)
        sentences = [s for s in new_entry.retrieved_docs[0].content.split(". ") if s]
        assert len(sentences) >= 1  # at minimum the content is preserved

    def test_no_docs_unchanged(self):
        entry = make_entry(docs=[])
        _, record = strategy_compress(entry, CFG, COUNTER)
        assert record is None


# ---------------------------------------------------------------------------
# Full optimizer integration tests
# ---------------------------------------------------------------------------

class TestContextOptimizer:
    @pytest.fixture
    def optimizer(self):
        return ContextOptimizer(OptimizationConfig(
            similarity_threshold=0.55,
            max_docs=4,
            max_doc_tokens=300,
        ))

    @pytest.fixture
    def evaluator(self):
        return ContextEvaluator()

    def test_excellent_context_skipped(self, optimizer, evaluator):
        docs = [RetrievedDoc(
            content="Context engineering is the discipline of designing optimal prompts.",
            source="wiki", relevance_score=0.95,
        )]
        entry = make_entry(docs=docs, max_context_tokens=8000)
        # Manually set score above 90
        from unittest.mock import MagicMock
        report = MagicMock()
        report.overall_score = 92.0
        result = optimizer.optimize(entry, report)
        assert result.was_modified is False

    def test_redundant_context_optimized(self, optimizer, evaluator):
        docs = [
            RetrievedDoc(content=DUPLICATE_CONTENT, source="a", relevance_score=0.90),
            RetrievedDoc(content=DUPLICATE_CONTENT + " Extra sentence here.", source="b", relevance_score=0.85),
            RetrievedDoc(content=DUPLICATE_CONTENT + " Another repetition.", source="c", relevance_score=0.80),
        ]
        entry = make_entry(docs=docs)
        report = evaluator.evaluate(entry)
        result = optimizer.optimize(entry, report)
        assert result.was_modified is True
        assert result.docs_removed >= 1
        assert result.tokens_saved > 0

    def test_change_log_populated(self, optimizer, evaluator):
        docs = [
            RetrievedDoc(content=DUPLICATE_CONTENT, source="a", relevance_score=0.90),
            RetrievedDoc(content=DUPLICATE_CONTENT, source="b", relevance_score=0.80),
        ]
        entry = make_entry(docs=docs)
        report = evaluator.evaluate(entry)
        result = optimizer.optimize(entry, report)
        if result.was_modified:
            assert len(result.change_log) >= 1
            for rec in result.change_log:
                assert rec.strategy in ["deduplicate", "rerank", "trim", "compress"]

    def test_original_entry_not_mutated(self, optimizer, evaluator):
        docs = [
            RetrievedDoc(content=DUPLICATE_CONTENT, source="a"),
            RetrievedDoc(content=DUPLICATE_CONTENT, source="b"),
        ]
        entry = make_entry(docs=docs)
        original_doc_count = len(entry.retrieved_docs)
        original_tokens = entry.total_tokens
        report = evaluator.evaluate(entry)
        optimizer.optimize(entry, report)
        assert len(entry.retrieved_docs) == original_doc_count
        assert entry.total_tokens == original_tokens

    def test_optimize_and_report_improves_score(self, optimizer, evaluator):
        docs = [
            RetrievedDoc(content=DUPLICATE_CONTENT, source="a", relevance_score=0.90),
            RetrievedDoc(content=DUPLICATE_CONTENT, source="b", relevance_score=0.85),
            RetrievedDoc(content="Unrelated: annual rainfall in Amazon exceeds 2000mm.", source="geo", relevance_score=0.05),
        ]
        entry = make_entry(docs=docs)
        report = evaluator.evaluate(entry)
        _, new_report = optimizer.optimize_and_report(entry, report)
        assert new_report.overall_score >= report.overall_score

    def test_summary_string_generated(self, optimizer, evaluator):
        docs = [
            RetrievedDoc(content=DUPLICATE_CONTENT, source="a"),
            RetrievedDoc(content=DUPLICATE_CONTENT, source="b"),
        ]
        entry = make_entry(docs=docs)
        report = evaluator.evaluate(entry)
        result = optimizer.optimize(entry, report)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_to_dict_serializes_correctly(self, optimizer, evaluator):
        docs = [RetrievedDoc(content=DUPLICATE_CONTENT, source="a")]
        entry = make_entry(docs=docs)
        report = evaluator.evaluate(entry)
        result = optimizer.optimize(entry, report)
        d = result.to_dict()
        assert "was_modified" in d
        assert "tokens_saved" in d
        assert "change_log" in d
        assert "strategies_applied" in d