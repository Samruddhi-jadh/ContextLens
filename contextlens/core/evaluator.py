# contextlens/core/evaluator.py
"""
Context Quality Evaluator — Phase 4

Scores a ContextEntry across 5 dimensions and produces an
EvaluationReport with a 0-100 grade, per-issue breakdown,
and concrete fix suggestions.

Design philosophy:
  - Each dimension is a pure function: (ContextEntry) -> DimensionScore
  - No side effects, no I/O — easy to test and easy to extend
  - Scoring is deterministic — same input always yields same score
  - Weights are configurable so users can tune for their use case

Scoring dimensions:
  1. Length       (weight 0.20) — is token usage healthy?
  2. Redundancy   (weight 0.25) — are docs repeating each other?
  3. Relevance    (weight 0.25) — do docs actually relate to the prompt?
  4. Specificity  (weight 0.15) — is the prompt precise enough?
  5. Completeness (weight 0.15) — are structural elements present?
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from loguru import logger

from contextlens.core.logger import ContextEntry


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """
    Score for one evaluation dimension.

    score:       0–100 (100 = perfect, 0 = critical problem)
    weight:      contribution to overall score (all weights sum to 1.0)
    issues:      list of specific problems found
    suggestions: actionable fixes for each issue
    detail:      numeric breakdown dict (shown in verbose/dashboard mode)
    """
    name: str
    score: float                          # 0–100
    weight: float                         # 0.0–1.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight

    @property
    def grade_letter(self) -> str:
        if self.score >= 90: return "A"
        if self.score >= 80: return "B"
        if self.score >= 70: return "C"
        if self.score >= 55: return "D"
        return "F"


@dataclass
class EvaluationReport:
    """
    Complete evaluation of one ContextEntry.

    This is the output of ContextEvaluator.evaluate().
    Downstream: optimizer reads issues + suggestions to fix context.
    CLI reads overall_score + grade for display.
    Dashboard reads dimension_scores for radar chart (Phase 10).
    """
    run_id: str
    overall_score: float                  # 0–100 weighted average
    grade: str                            # A/B/C/D/F
    dimension_scores: list[DimensionScore]
    all_issues: list[str]                 # flattened across all dimensions
    all_suggestions: list[str]            # flattened, deduplicated
    token_stats: dict[str, float]         # raw numbers for reference
    passed: bool                          # True if overall_score >= 70

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "overall_score": round(self.overall_score, 2),
            "grade": self.grade,
            "passed": self.passed,
            "dimensions": {
                ds.name: {
                    "score": round(ds.score, 2),
                    "weight": ds.weight,
                    "grade": ds.grade_letter,
                    "issues": ds.issues,
                    "suggestions": ds.suggestions,
                    "detail": {k: round(v, 4) for k, v in ds.detail.items()},
                }
                for ds in self.dimension_scores
            },
            "all_issues": self.all_issues,
            "all_suggestions": self.all_suggestions,
            "token_stats": {k: round(v, 4) for k, v in self.token_stats.items()},
        }


# ---------------------------------------------------------------------------
# Dimension scorer functions
# ---------------------------------------------------------------------------

def score_length(entry: ContextEntry) -> DimensionScore:
    """
    Dimension 1: Length / token budget health.

    Scoring rationale:
      - <20% utilization → context likely too sparse → score 60
        (you're paying for a big context window but barely using it)
      - 20–80% → ideal range → score 100 → 80 (linear drop after 70%)
      - 80–90% → getting tight → score 70 → 50
      - 90–100% → danger zone → score 50 → 10
      - >100% → overflow → score 0

    We also penalize when docs dominate (>60% of total tokens)
    because that usually means you retrieved too many low-quality docs.
    """
    util = entry.token_utilization          # 0.0 – 1.0+
    total = entry.total_tokens
    max_t = entry.max_context_tokens
    issues = []
    suggestions = []

    # Base score from utilization curve
    if util > 1.0:
        base = 0.0
        issues.append(
            f"Context OVERFLOW: {total} tokens exceeds limit of {max_t}. "
            "API call will fail."
        )
        suggestions.append(
            "Immediately run optimizer to trim context below the token limit."
        )
    elif util > 0.90:
        base = max(10.0, 50.0 - (util - 0.90) * 400)
        issues.append(
            f"Critical: context at {util:.1%} capacity ({total}/{max_t} tokens). "
            "High risk of truncation."
        )
        suggestions.append("Remove lowest-relevance retrieved docs first.")
    elif util > 0.80:
        base = max(50.0, 70.0 - (util - 0.80) * 200)
        issues.append(f"Context at {util:.1%} capacity — approaching limit.")
        suggestions.append("Consider trimming retrieved docs by 20%.")
    elif util < 0.20 and total > 0:
        base = 60.0
        issues.append(
            f"Sparse context: only {util:.1%} of token budget used ({total} tokens). "
            "Consider adding more retrieved context."
        )
        suggestions.append(
            "Retrieve 2–3 more relevant documents to give the model richer context."
        )
    else:
        # Sweet spot: 20–80% — score 100 at 50%, gentle linear drop
        base = 100.0 - max(0.0, (util - 0.50) * 100)

    # Penalty: docs dominating (>60% of total tokens means overstuffed RAG)
    doc_ratio = entry.docs_tokens / max(1, total)
    if doc_ratio > 0.60 and entry.retrieved_docs:
        penalty = (doc_ratio - 0.60) * 50
        base = max(0.0, base - penalty)
        issues.append(
            f"Retrieved docs consume {doc_ratio:.1%} of context. "
            f"Prompt and system message are being squeezed out."
        )
        suggestions.append(
            "Reduce number of retrieved docs or apply content trimming per doc."
        )

    return DimensionScore(
        name="length",
        score=round(min(100.0, max(0.0, base)), 2),
        weight=0.20,
        issues=issues,
        suggestions=suggestions,
        detail={
            "token_utilization": util,
            "total_tokens": float(total),
            "max_tokens": float(max_t),
            "doc_ratio": doc_ratio,
        },
    )


def score_redundancy(entry: ContextEntry) -> DimensionScore:
    """
    Dimension 2: Redundancy across retrieved documents.

    Method: pairwise cosine similarity using TF-IDF-style term vectors.
    We avoid importing sentence-transformers (heavy) and instead use
    a lightweight term-frequency approach that's fast and good enough
    for detecting obvious duplicate content.

    Scoring:
      - 0 docs or 1 doc → perfect score (no redundancy possible)
      - avg pairwise similarity > 0.7 → severe redundancy
      - avg pairwise similarity > 0.4 → moderate redundancy
      - avg pairwise similarity < 0.2 → diverse, excellent
    """
    docs = entry.retrieved_docs
    issues = []
    suggestions = []

    if len(docs) <= 1:
        return DimensionScore(
            name="redundancy",
            score=100.0,
            weight=0.25,
            detail={"doc_count": float(len(docs)), "avg_similarity": 0.0},
        )

    # Build term-frequency vectors for each doc
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z]{3,}\b", text.lower())

    def tf_vector(tokens: list[str]) -> dict[str, float]:
        freq: dict[str, float] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = max(1, len(tokens))
        return {t: c / total for t, c in freq.items()}

    def cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        norm_a = math.sqrt(sum(v**2 for v in a.values()))
        norm_b = math.sqrt(sum(v**2 for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    vectors = [tf_vector(tokenize(doc.content)) for doc in docs]

    # Pairwise similarities
    similarities = []
    redundant_pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_sim(vectors[i], vectors[j])
            similarities.append(sim)
            if sim > 0.55:
                redundant_pairs.append((i, j, sim))

    avg_sim = sum(similarities) / max(1, len(similarities))
    max_sim = max(similarities) if similarities else 0.0

    # Score from avg similarity
    if avg_sim > 0.70:
        score = max(10.0, 30.0 - (avg_sim - 0.70) * 100)
        issues.append(
            f"Severe redundancy: avg document similarity is {avg_sim:.2f}. "
            f"Documents are near-duplicates."
        )
        suggestions.append(
            "Deduplicate retrieved docs — keep only the highest-relevance "
            "version of similar content."
        )
    elif avg_sim > 0.40:
        score = max(50.0, 70.0 - (avg_sim - 0.40) * 80)
        issues.append(
            f"Moderate redundancy: avg similarity {avg_sim:.2f}. "
            f"{len(redundant_pairs)} overlapping pairs found."
        )
        suggestions.append(
            "Review and merge overlapping retrieved documents before passing to context."
        )
    elif avg_sim > 0.20:
        score = 85.0 - (avg_sim - 0.20) * 75
        # Mild redundancy — note it but don't penalize heavily
        if redundant_pairs:
            issues.append(f"Mild overlap detected in {len(redundant_pairs)} doc pairs.")
            suggestions.append("Consider light deduplication for cleaner context.")
    else:
        score = 100.0  # great diversity

    return DimensionScore(
        name="redundancy",
        score=round(score, 2),
        weight=0.25,
        issues=issues,
        suggestions=suggestions,
        detail={
            "doc_count": float(len(docs)),
            "avg_similarity": round(avg_sim, 4),
            "max_similarity": round(max_sim, 4),
            "redundant_pairs": float(len(redundant_pairs)),
        },
    )


def score_relevance(entry: ContextEntry) -> DimensionScore:
    """
    Dimension 3: Relevance of retrieved docs to the user prompt.

    Method: term overlap between user_prompt keywords and doc content.
    We extract meaningful tokens (nouns, verbs — anything 4+ chars
    that isn't a stopword) from the prompt and check what % appear
    in each doc.

    Also factors in retriever-provided relevance_score when available.

    Scoring:
      - If retriever scores available: blend with term overlap 50/50
      - If no docs: score 70 (neutral — no docs isn't inherently bad)
      - Low term overlap (<20%) → docs don't match the question
    """
    docs = entry.retrieved_docs
    prompt = entry.user_prompt
    issues = []
    suggestions = []

    if not docs:
        return DimensionScore(
            name="relevance",
            score=70.0,
            weight=0.25,
            issues=["No retrieved documents in context."],
            suggestions=[
                "If this is a knowledge-intensive task, add retrieved docs via RAG."
            ],
            detail={"doc_count": 0.0},
        )

    # Extract prompt keywords (4+ chars, not stopwords)
    STOPWORDS = {
        "that", "this", "with", "from", "have", "will", "been", "were",
        "they", "what", "when", "where", "which", "there", "their",
        "about", "would", "could", "should", "more", "into", "also",
        "some", "than", "then", "them", "these", "those", "your",
    }
    prompt_tokens = set(
        t for t in re.findall(r"\b[a-z]{4,}\b", prompt.lower())
        if t not in STOPWORDS
    )

    if not prompt_tokens:
        return DimensionScore(
            name="relevance",
            score=60.0,
            weight=0.25,
            issues=["User prompt is too short to extract meaningful keywords."],
            suggestions=["Make the user prompt more specific and detailed."],
            detail={"prompt_keywords": 0.0},
        )

    # Term overlap per doc
    doc_overlap_scores = []
    for doc in docs:
        doc_tokens = set(re.findall(r"\b[a-z]{4,}\b", doc.content.lower()))
        overlap = len(prompt_tokens & doc_tokens) / len(prompt_tokens)
        doc_overlap_scores.append(overlap)

    avg_overlap = sum(doc_overlap_scores) / len(doc_overlap_scores)

    # Blend with retriever-provided scores if available
    retriever_scores = [
        doc.relevance_score for doc in docs
        if doc.relevance_score is not None
    ]
    if retriever_scores:
        avg_retriever = sum(retriever_scores) / len(retriever_scores)
        # 50% term overlap, 50% retriever score
        combined = (avg_overlap * 0.5) + (avg_retriever * 0.5)
    else:
        combined = avg_overlap

    # Score from combined relevance
    score = min(100.0, combined * 120)  # scale: 0.83 overlap → 100

    if combined < 0.25:
        issues.append(
            f"Low relevance: docs share only {avg_overlap:.1%} keyword overlap "
            "with the user prompt. Retrieved content may not be on-topic."
        )
        suggestions.append(
            "Revisit your retrieval query — consider using the user prompt "
            "directly as the embedding search query."
        )
    elif combined < 0.45:
        issues.append(
            f"Moderate relevance: {avg_overlap:.1%} keyword overlap with prompt."
        )
        suggestions.append(
            "Refine retrieval to fetch docs more closely matching prompt keywords."
        )

    # Flag docs with low individual relevance
    low_relevance_docs = [
        i for i, s in enumerate(doc_overlap_scores) if s < 0.15
    ]
    if low_relevance_docs:
        sources = [docs[i].source for i in low_relevance_docs]
        issues.append(
            f"Docs from [{', '.join(sources)}] have low relevance to the prompt."
        )
        suggestions.append(
            f"Consider removing or replacing docs from: {', '.join(sources)}."
        )

    return DimensionScore(
        name="relevance",
        score=round(max(0.0, score), 2),
        weight=0.25,
        issues=issues,
        suggestions=suggestions,
        detail={
            "prompt_keywords": float(len(prompt_tokens)),
            "avg_term_overlap": round(avg_overlap, 4),
            "avg_retriever_score": round(
                sum(retriever_scores) / max(1, len(retriever_scores)), 4
            ) if retriever_scores else -1.0,
            "combined_relevance": round(combined, 4),
            "low_relevance_doc_count": float(len(low_relevance_docs)),
        },
    )


def score_specificity(entry: ContextEntry) -> DimensionScore:
    """
    Dimension 4: Prompt specificity — is the question precise?

    Vague prompts cause hallucinations even with perfect context.
    We detect vagueness through several signals:

    Penalty signals (each deducts points):
      - Very short prompt (<8 tokens) → probably too vague
      - Vague filler words: "something", "stuff", "things", "basically"
      - No question structure: no "?", "how", "what", "why", "explain"
      - All-lowercase, no punctuation → casual/underspecified
      - Overly long prompt (>200 words) → may be unfocused

    Bonus signals:
      - Contains specific nouns (names, technical terms, numbers)
      - Has explicit constraint words: "in Python", "in 3 steps", "as JSON"
      - Has a clear question word
    """
    prompt = entry.user_prompt
    issues = []
    suggestions = []

    words = prompt.split()
    word_count = len(words)
    score = 70.0  # neutral start

    VAGUE_WORDS = {
        "something", "stuff", "things", "basically", "kinda", "sorta",
        "maybe", "perhaps", "somehow", "whatever", "everything", "anything",
        "general", "generally", "usually", "various", "some", "certain",
    }
    QUESTION_WORDS = {
        "how", "what", "why", "when", "where", "which", "who",
        "explain", "describe", "compare", "list", "summarize", "analyze",
        "implement", "build", "write", "create", "design",
    }
    CONSTRAINT_PATTERNS = [
        r"\bin\s+(python|javascript|java|sql|rust|go|typescript)\b",
        r"\bin\s+\d+\s+(steps?|words?|sentences?|paragraphs?|lines?)\b",
        r"\bas\s+(json|xml|yaml|csv|markdown|a list|bullet points?)\b",
        r"\bfor\s+(production|beginners?|experts?|enterprise)\b",
        r"\bwithout\s+\w+",
        r"\busing\s+\w+",
        r"\bmax\s+\d+",
        r"\bat\s+least\s+\d+",
    ]

    prompt_lower = prompt.lower()

    # Length checks
    if word_count < 5:
        score -= 30
        issues.append(
            f"Prompt is very short ({word_count} words). "
            "Too vague for reliable results."
        )
        suggestions.append(
            "Expand the prompt: add context, specify the desired format, "
            "or clarify the scope."
        )
    elif word_count < 10:
        score -= 15
        issues.append(f"Short prompt ({word_count} words) — likely underspecified.")
        suggestions.append("Add more detail: what format, what constraints, what level of detail?")
    elif word_count > 200:
        score -= 10
        issues.append(
            f"Very long prompt ({word_count} words) may be unfocused. "
            "Consider breaking into sub-prompts."
        )
        suggestions.append("Split into a clear question + separate context/constraints.")

    # Vague word penalty
    vague_found = [w for w in words if w.lower().rstrip(".,?!") in VAGUE_WORDS]
    if vague_found:
        penalty = min(20, len(vague_found) * 7)
        score -= penalty
        issues.append(
            f"Vague language detected: {vague_found[:4]}. "
            "Imprecise prompts increase hallucination risk."
        )
        suggestions.append(
            "Replace vague words with specific terms. "
            "Instead of 'explain something about X', use 'explain how X works in Y context'."
        )

    # Question structure bonus
    has_question_word = any(w in prompt_lower for w in QUESTION_WORDS)
    if has_question_word:
        score += 10
    else:
        issues.append("No clear task directive (how/what/why/explain/build/write...).")
        suggestions.append(
            "Start the prompt with a clear action: "
            "'Explain...', 'Write...', 'Compare...', 'List...'"
        )

    # Constraint bonus
    constraints_found = sum(
        1 for p in CONSTRAINT_PATTERNS
        if re.search(p, prompt_lower)
    )
    if constraints_found >= 2:
        score += 15
    elif constraints_found == 1:
        score += 8
    else:
        suggestions.append(
            "Add constraints for better results: format (JSON/markdown), "
            "length, language, audience level, or use case."
        )

    # Specific noun/number bonus (signals technical precision)
    has_numbers = bool(re.search(r"\b\d+\b", prompt))
    has_caps = bool(re.search(r"\b[A-Z][a-z]{2,}\b", prompt))  # proper nouns
    if has_numbers:
        score += 5
    if has_caps:
        score += 5

    return DimensionScore(
        name="specificity",
        score=round(min(100.0, max(0.0, score)), 2),
        weight=0.15,
        issues=issues,
        suggestions=suggestions,
        detail={
            "word_count": float(word_count),
            "vague_word_count": float(len(vague_found)),
            "has_question_word": float(has_question_word),
            "constraints_found": float(constraints_found),
        },
    )


def score_completeness(entry: ContextEntry) -> DimensionScore:
    """
    Dimension 5: Structural completeness of the context.

    Checks for the presence of important context elements.
    Missing elements don't always cause problems, but they're
    common sources of inconsistent agent behaviour.

    Checks:
      - System prompt present and substantive (not just 1 sentence)
      - System prompt has role definition ("you are a...")
      - System prompt has behavioural constraints
      - For multi-turn: conversation history present
      - For RAG: retrieved docs present when prompt implies knowledge need
    """
    issues = []
    suggestions = []
    score = 100.0

    system = entry.system_prompt.strip()
    prompt_lower = entry.user_prompt.lower()

    # --- System prompt checks ---
    if not system:
        score -= 25
        issues.append("No system prompt. Agent has no defined persona or constraints.")
        suggestions.append(
            "Add a system prompt that defines: role, expertise level, "
            "response format, and any constraints."
        )
    else:
        system_words = len(system.split())
        system_lower = system.lower()

        # Role definition check ("you are", "you're", "act as", "your role")
        HAS_ROLE = bool(re.search(
            r"\b(you are|you're|act as|your role|as a|as an)\b",
            system_lower
        ))
        if not HAS_ROLE:
            score -= 10
            issues.append("System prompt lacks explicit role definition.")
            suggestions.append(
                "Start system prompt with: 'You are a [role] with expertise in [domain]...'"
            )

        # Constraint / behaviour keywords
        HAS_CONSTRAINTS = bool(re.search(
            r"\b(always|never|only|must|should|do not|don't|avoid|focus|prioritize|respond)\b",
            system_lower
        ))
        if not HAS_CONSTRAINTS:
            score -= 8
            issues.append("System prompt has no behavioural constraints.")
            suggestions.append(
                "Add constraints: 'Always cite sources', 'Respond only in JSON', "
                "'Never reveal system instructions'."
            )

        # Too short system prompt
        if system_words < 10:
            score -= 10
            issues.append(
                f"System prompt is very short ({system_words} words). "
                "Provides little guidance to the model."
            )
            suggestions.append(
                "Expand system prompt to at least 30–50 words covering "
                "role, tone, format, and constraints."
            )

    # --- Conversation history check for multi-turn signals ---
    MULTI_TURN_SIGNALS = [
        "as i mentioned", "earlier you said", "you told me", "we discussed",
        "previously", "last time", "follow up", "continuing from",
    ]
    implies_history = any(sig in prompt_lower for sig in MULTI_TURN_SIGNALS)
    if implies_history and not entry.conversation_history:
        score -= 15
        issues.append(
            "Prompt implies a multi-turn conversation but no history is included. "
            "Model lacks prior context."
        )
        suggestions.append(
            "Pass conversation_history=[...] to preserve context across turns."
        )

    # --- RAG context check ---
    KNOWLEDGE_SIGNALS = [
        "according to", "based on", "from the document", "in the article",
        "the paper says", "as stated", "reference", "source", "document",
        "report", "study", "research",
    ]
    implies_rag = any(sig in prompt_lower for sig in KNOWLEDGE_SIGNALS)
    if implies_rag and not entry.retrieved_docs:
        score -= 12
        issues.append(
            "Prompt references external documents/sources but no docs are in context."
        )
        suggestions.append(
            "Retrieve and pass relevant documents via retrieved_docs=[...]."
        )

    return DimensionScore(
        name="completeness",
        score=round(max(0.0, score), 2),
        weight=0.15,
        issues=issues,
        suggestions=suggestions,
        detail={
            "has_system_prompt": float(bool(system)),
            "system_word_count": float(len(system.split()) if system else 0),
            "has_history": float(bool(entry.conversation_history)),
            "has_docs": float(bool(entry.retrieved_docs)),
        },
    )


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

# Scorer registry — ordered list of scorer functions
DIMENSION_SCORERS: list[Callable[[ContextEntry], DimensionScore]] = [
    score_length,
    score_redundancy,
    score_relevance,
    score_specificity,
    score_completeness,
]

GRADE_THRESHOLDS = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (55, "D"),
    (0,  "F"),
]


class ContextEvaluator:
    """
    Evaluates context quality across 5 dimensions.

    Usage:
        evaluator = ContextEvaluator()
        report = evaluator.evaluate(context_entry)
        print(report.overall_score)   # e.g. 74.5
        print(report.grade)           # e.g. "C"
        print(report.all_issues)      # list of problems
        print(report.all_suggestions) # list of fixes
    """

    def __init__(
        self,
        scorers: list[Callable] | None = None,
    ):
        """
        Args:
            scorers: Custom scorer list. Defaults to all 5 built-in scorers.
                     Inject custom scorers here for domain-specific evaluation.
        """
        self._scorers = scorers or DIMENSION_SCORERS

    def evaluate(self, entry: ContextEntry) -> EvaluationReport:
        """
        Run all dimension scorers and aggregate into an EvaluationReport.

        Returns:
            EvaluationReport with overall score, grade, issues, suggestions.
        """
        dimension_scores: list[DimensionScore] = []
        for scorer in self._scorers:
            try:
                ds = scorer(entry)
                dimension_scores.append(ds)
            except Exception as e:
                logger.error(
                    "Scorer '{}' failed on run_id={}: {}",
                    scorer.__name__, entry.run_id[:8], e
                )
                # Scorer failure = neutral score, never crash
                dimension_scores.append(DimensionScore(
                    name=scorer.__name__.replace("score_", ""),
                    score=50.0,
                    weight=0.0,
                    issues=[f"Scorer error: {e}"],
                ))

        # Weighted average
        total_weight = sum(ds.weight for ds in dimension_scores)
        if total_weight == 0:
            overall = 50.0
        else:
            overall = sum(ds.weighted_score for ds in dimension_scores) / total_weight

        overall = round(min(100.0, max(0.0, overall)), 2)

        # Grade
        grade = "F"
        for threshold, letter in GRADE_THRESHOLDS:
            if overall >= threshold:
                grade = letter
                break

        # Flatten issues and suggestions (deduplicated)
        seen_suggestions: set[str] = set()
        all_issues: list[str] = []
        all_suggestions: list[str] = []
        for ds in dimension_scores:
            all_issues.extend(ds.issues)
            for s in ds.suggestions:
                if s not in seen_suggestions:
                    all_suggestions.append(s)
                    seen_suggestions.add(s)

        token_stats = {
            "total_tokens": float(entry.total_tokens),
            "system_tokens": float(entry.system_tokens),
            "user_tokens": float(entry.user_tokens),
            "docs_tokens": float(entry.docs_tokens),
            "history_tokens": float(entry.history_tokens),
            "token_utilization": entry.token_utilization,
        }

        report = EvaluationReport(
            run_id=entry.run_id,
            overall_score=overall,
            grade=grade,
            dimension_scores=dimension_scores,
            all_issues=all_issues,
            all_suggestions=all_suggestions,
            token_stats=token_stats,
            passed=overall >= 70.0,
        )

        logger.info(
            "Evaluation complete | run_id={} | score={}/100 | grade={} | issues={}",
            entry.run_id[:8],
            overall,
            grade,
            len(all_issues),
        )

        return report