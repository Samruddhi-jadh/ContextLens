# tests/test_evaluator.py
"""
Tests for the ContextEvaluator and all 5 dimension scorers.

Strategy: test each scorer in isolation with carefully constructed
ContextEntry fixtures. Then test the full evaluator end-to-end.
This is the most important test file in the project — if scoring
is wrong, all downstream features (optimizer, CLI) are wrong too.
"""

import pytest
from contextlens.core.logger import ContextEntry, RetrievedDoc
from contextlens.core.evaluator import (
    ContextEvaluator,
    score_length,
    score_redundancy,
    score_relevance,
    score_specificity,
    score_completeness,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_entry(**kwargs) -> ContextEntry:
    """Helper — build a ContextEntry with sensible defaults."""
    defaults = dict(
        user_prompt="What is context engineering and why does it matter?",
        system_prompt="You are a senior AI engineer. Be precise and concise.",
        retrieved_docs=[],
        total_tokens=300,
        system_tokens=12,
        user_tokens=10,
        docs_tokens=0,
        history_tokens=0,
        max_context_tokens=8000,
        provider="groq",
        model="llama-3.1-70b-versatile",
    )
    defaults.update(kwargs)
    return ContextEntry(**defaults)


# ---------------------------------------------------------------------------
# Dimension 1: Length scorer
# ---------------------------------------------------------------------------

class TestScoreLength:
    def test_ideal_utilization_scores_high(self):
        entry = make_entry(total_tokens=2000, max_context_tokens=8000)  # 25%
        ds = score_length(entry)
        assert ds.score >= 85

    def test_overflow_scores_zero(self):
        entry = make_entry(total_tokens=9000, max_context_tokens=8000)
        ds = score_length(entry)
        assert ds.score == 0.0
        assert any("OVERFLOW" in i for i in ds.issues)

    def test_near_limit_scores_low(self):
        entry = make_entry(total_tokens=7500, max_context_tokens=8000)  # 93.75%
        ds = score_length(entry)
        assert ds.score < 50

    def test_sparse_context_flagged(self):
        entry = make_entry(total_tokens=100, max_context_tokens=8000)  # 1.25%
        ds = score_length(entry)
        assert ds.score <= 65
        assert any("Sparse" in i or "sparse" in i for i in ds.issues)

    def test_weight_is_correct(self):
        entry = make_entry(total_tokens=1000, max_context_tokens=8000)
        ds = score_length(entry)
        assert ds.weight == pytest.approx(0.20)

    def test_doc_dominated_context_penalized(self):
        entry = make_entry(
            total_tokens=1000,
            docs_tokens=800,    # 80% docs — should be penalized
            max_context_tokens=8000,
        )
        entry.retrieved_docs = [RetrievedDoc(content="x" * 100)]
        ds = score_length(entry)
        assert any("doc" in i.lower() for i in ds.issues)


# ---------------------------------------------------------------------------
# Dimension 2: Redundancy scorer
# ---------------------------------------------------------------------------

class TestScoreRedundancy:
    def test_single_doc_perfect_score(self):
        doc = RetrievedDoc(content="Context engineering is important for LLMs.")
        entry = make_entry(retrieved_docs=[doc])
        ds = score_redundancy(entry)
        assert ds.score == 100.0

    def test_no_docs_perfect_score(self):
        entry = make_entry(retrieved_docs=[])
        ds = score_redundancy(entry)
        assert ds.score == 100.0

    def test_near_duplicate_docs_score_low(self):
        content = (
            "Context engineering is the practice of designing and optimizing "
            "the information passed to large language models for better results."
        )
        doc1 = RetrievedDoc(content=content)
        doc2 = RetrievedDoc(content=content + " This is important for production AI.")
        entry = make_entry(retrieved_docs=[doc1, doc2])
        ds = score_redundancy(entry)
        assert ds.score < 60
        assert len(ds.issues) > 0

    def test_diverse_docs_score_high(self):
        doc1 = RetrievedDoc(content="Quantum physics explores subatomic particle behaviour.")
        doc2 = RetrievedDoc(content="Machine learning models train on labelled datasets.")
        doc3 = RetrievedDoc(content="Ocean currents distribute heat around the planet.")
        entry = make_entry(retrieved_docs=[doc1, doc2, doc3])
        ds = score_redundancy(entry)
        assert ds.score >= 85

    def test_weight_is_correct(self):
        entry = make_entry(retrieved_docs=[])
        ds = score_redundancy(entry)
        assert ds.weight == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Dimension 3: Relevance scorer
# ---------------------------------------------------------------------------

class TestScoreRelevance:
    def test_highly_relevant_docs_score_high(self):
        doc = RetrievedDoc(
            content=(
                "Context engineering involves designing optimal prompts, "
                "retrieving relevant documents, and managing token budgets "
                "to improve language model performance."
            ),
            source="wiki",
            relevance_score=0.95,
        )
        entry = make_entry(
            user_prompt="What is context engineering and how does it improve language models?",
            retrieved_docs=[doc],
        )
        ds = score_relevance(entry)
        assert ds.score >= 70

    def test_irrelevant_docs_score_low(self):
        doc = RetrievedDoc(
            content="The annual rainfall in the Amazon basin exceeds 2000mm per year.",
            source="geography_db",
            relevance_score=0.1,
        )
        entry = make_entry(
            user_prompt="How do transformer attention mechanisms work?",
            retrieved_docs=[doc],
        )
        ds = score_relevance(entry)
        assert ds.score < 50

    def test_no_docs_returns_neutral_score(self):
        entry = make_entry(retrieved_docs=[])
        ds = score_relevance(entry)
        assert ds.score == pytest.approx(70.0)

    def test_weight_is_correct(self):
        entry = make_entry(retrieved_docs=[])
        ds = score_relevance(entry)
        assert ds.weight == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Dimension 4: Specificity scorer
# ---------------------------------------------------------------------------

class TestScoreSpecificity:
    def test_vague_prompt_scores_low(self):
        entry = make_entry(user_prompt="explain something about AI stuff")
        ds = score_specificity(entry)
        assert ds.score < 60
        assert any("ague" in i for i in ds.issues)

    def test_specific_constrained_prompt_scores_high(self):
        entry = make_entry(
            user_prompt=(
                "Explain how the Transformer attention mechanism works in Python, "
                "using a concrete example with at least 3 steps, "
                "for a senior ML engineer audience."
            )
        )
        ds = score_specificity(entry)
        assert ds.score >= 80

    def test_very_short_prompt_penalized(self):
        entry = make_entry(user_prompt="Tell me stuff")
        ds = score_specificity(entry)
        assert ds.score < 55

    def test_question_word_gives_bonus(self):
        with_q = make_entry(user_prompt="How does context engineering work in production systems?")
        without_q = make_entry(user_prompt="context engineering production systems information")
        ds_with = score_specificity(with_q)
        ds_without = score_specificity(without_q)
        assert ds_with.score > ds_without.score

    def test_weight_is_correct(self):
        entry = make_entry()
        ds = score_specificity(entry)
        assert ds.weight == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Dimension 5: Completeness scorer
# ---------------------------------------------------------------------------

class TestScoreCompleteness:
    def test_no_system_prompt_penalized(self):
        entry = make_entry(system_prompt="")
        ds = score_completeness(entry)
        assert ds.score <= 75
        assert any("system prompt" in i.lower() for i in ds.issues)

    def test_rich_system_prompt_scores_high(self):
        entry = make_entry(
            system_prompt=(
                "You are a senior AI engineer with 10 years of experience in "
                "production LLM systems. Always respond with precise technical "
                "detail. Never speculate. Format responses as structured markdown."
            )
        )
        ds = score_completeness(entry)
        assert ds.score >= 90

    def test_system_prompt_without_role_loses_points(self):
        entry = make_entry(system_prompt="Be helpful and accurate.")
        ds = score_completeness(entry)
        assert any("role" in i.lower() for i in ds.issues)

    def test_multi_turn_signal_without_history_flagged(self):
        entry = make_entry(
            user_prompt="As I mentioned earlier, I need help with context engineering.",
            conversation_history=[],
        )
        ds = score_completeness(entry)
        assert any("history" in i.lower() or "multi-turn" in i.lower() for i in ds.issues)

    def test_weight_is_correct(self):
        entry = make_entry()
        ds = score_completeness(entry)
        assert ds.weight == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Full evaluator integration tests
# ---------------------------------------------------------------------------

class TestContextEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ContextEvaluator()

    def test_good_context_scores_above_70(self, evaluator):
        docs = [
            RetrievedDoc(
                content=(
                    "Context engineering involves designing optimal context windows "
                    "for large language models to maximize accuracy and minimize cost."
                ),
                source="wiki",
                relevance_score=0.92,
            )
        ]
        entry = make_entry(
            user_prompt=(
                "Explain how context engineering improves language model accuracy "
                "in production systems, with at least 2 concrete examples."
            ),
            system_prompt=(
                "You are a senior AI engineer. Always respond with technical precision. "
                "Cite your sources. Never speculate beyond the provided context."
            ),
            retrieved_docs=docs,
            total_tokens=400,
            docs_tokens=120,
            max_context_tokens=8000,
        )
        report = evaluator.evaluate(entry)
        assert report.overall_score >= 70
        assert report.passed is True

    def test_poor_context_scores_below_60(self, evaluator):
        content = "AI is important for many things in various different ways."
        doc1 = RetrievedDoc(content=content, source="a")
        doc2 = RetrievedDoc(content=content + " It does stuff.", source="b")
        entry = make_entry(
            user_prompt="tell me something about stuff",
            system_prompt="",
            retrieved_docs=[doc1, doc2],
            total_tokens=7800,
            docs_tokens=6000,
            max_context_tokens=8000,
        )
        report = evaluator.evaluate(entry)
        assert report.overall_score < 65
        assert len(report.all_issues) >= 3

    def test_report_has_all_dimensions(self, evaluator):
        entry = make_entry()
        report = evaluator.evaluate(entry)
        names = {ds.name for ds in report.dimension_scores}
        assert "length" in names
        assert "redundancy" in names
        assert "relevance" in names
        assert "specificity" in names
        assert "completeness" in names

    def test_report_serializes_to_dict(self, evaluator):
        entry = make_entry()
        report = evaluator.evaluate(entry)
        d = report.to_dict()
        assert "overall_score" in d
        assert "grade" in d
        assert "dimensions" in d
        assert "all_issues" in d

    def test_scorer_failure_does_not_crash(self, evaluator):
        """A broken scorer must not crash the whole evaluation."""
        def bad_scorer(entry):
            raise RuntimeError("Intentional failure")

        evaluator_with_bad = ContextEvaluator(scorers=[bad_scorer])
        entry = make_entry()
        report = evaluator_with_bad.evaluate(entry)  # must not raise
        assert report is not None