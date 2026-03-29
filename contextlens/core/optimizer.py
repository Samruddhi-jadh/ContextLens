# contextlens/core/optimizer.py
"""
Context Optimizer — Phase 6

Reads an EvaluationReport and automatically repairs the ContextEntry
by applying targeted strategies. Returns an OptimizationResult containing
the improved context and a full audit trail of what changed and why.

Four strategies (applied in this order — order matters):
  1. Deduplicate   — remove near-duplicate retrieved docs
  2. Re-rank       — sort docs by relevance, keep top-K
  3. Trim          — truncate long docs to fit token budget
  4. Compress      — filter sentences by keyword relevance

Design decisions:
  - Strategies are pure functions: (ContextEntry, config) → ContextEntry
  - They compose: each takes the output of the previous
  - OptimizationResult carries a full change_log for the CLI and dashboard
  - Original ContextEntry is never mutated (immutable input principle)
  - Token counter reused from Phase 2 (no duplicate logic)
  - Threshold values are configurable via OptimizationConfig
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from contextlens.core.evaluator import EvaluationReport
from contextlens.core.logger import ContextEntry, RetrievedDoc, TokenCounter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """
    Tunable thresholds for all optimization strategies.

    Expose these in the CLI and Streamlit dashboard (Phase 10)
    so users can tune for their use case without touching code.
    """
    # Deduplication
    similarity_threshold: float = 0.55
    """Docs with cosine similarity above this are considered duplicates."""

    # Re-ranking
    max_docs: int = 5
    """Maximum number of docs to keep after re-ranking."""

    min_relevance_score: float = 0.0
    """Docs with retriever relevance_score below this are dropped.
    0.0 = no filter (use re-ranking alone)."""

    # Trimming
    target_token_budget: float = 0.75
    """After trimming, docs should consume no more than this fraction
    of max_context_tokens. 0.75 = leave 25% for prompt + system msg."""

    max_doc_tokens: int = 400
    """Hard cap per individual document (tokens)."""

    # Compression
    enable_compression: bool = True
    """Whether to apply sentence-level compression."""

    compression_keyword_ratio: float = 0.4
    """Keep sentences where at least this fraction of prompt keywords appear."""

    min_sentences_per_doc: int = 2
    """Never compress a doc below this many sentences."""


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class ChangeRecord:
    """One atomic change made during optimization."""
    strategy: str           # "deduplicate" | "rerank" | "trim" | "compress"
    description: str        # human-readable explanation
    tokens_before: int
    tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass
class OptimizationResult:
    """
    Complete result of one optimization pass.

    Contains the improved ContextEntry plus a full audit trail.
    The audit trail is what makes this research-worthy — you can
    see exactly what changed, why, and how much was saved.
    """
    original_entry: ContextEntry
    optimized_entry: ContextEntry
    change_log: list[ChangeRecord] = field(default_factory=list)
    strategies_applied: list[str] = field(default_factory=list)
    was_modified: bool = False

    @property
    def tokens_saved(self) -> int:
        return (
            self.original_entry.total_tokens
            - self.optimized_entry.total_tokens
        )

    @property
    def docs_removed(self) -> int:
        return (
            len(self.original_entry.retrieved_docs)
            - len(self.optimized_entry.retrieved_docs)
        )

    @property
    def token_reduction_pct(self) -> float:
        orig = self.original_entry.total_tokens
        if orig == 0:
            return 0.0
        return (self.tokens_saved / orig) * 100

    def summary(self) -> str:
        if not self.was_modified:
            return "No optimization needed — context already healthy."
        lines = [
            f"Optimized: {self.tokens_saved:+d} tokens saved "
            f"({self.token_reduction_pct:.1f}% reduction)",
            f"Strategies: {', '.join(self.strategies_applied)}",
            f"Docs removed: {self.docs_removed}",
        ]
        for rec in self.change_log:
            lines.append(f"  [{rec.strategy}] {rec.description} "
                         f"(-{rec.tokens_saved} tokens)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "was_modified": self.was_modified,
            "tokens_saved": self.tokens_saved,
            "token_reduction_pct": round(self.token_reduction_pct, 2),
            "docs_removed": self.docs_removed,
            "strategies_applied": self.strategies_applied,
            "original_tokens": self.original_entry.total_tokens,
            "optimized_tokens": self.optimized_entry.total_tokens,
            "change_log": [
                {
                    "strategy": r.strategy,
                    "description": r.description,
                    "tokens_before": r.tokens_before,
                    "tokens_after": r.tokens_after,
                    "tokens_saved": r.tokens_saved,
                }
                for r in self.change_log
            ],
        }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """TF cosine similarity — reused from evaluator logic."""
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tf_vector(text: str) -> dict[str, float]:
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    freq: dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = max(1, len(tokens))
    return {t: c / total for t, c in freq.items()}


def _rebuild_entry(
    original: ContextEntry,
    new_docs: list[RetrievedDoc],
    counter: TokenCounter,
) -> ContextEntry:
    """
    Clone the original ContextEntry with a new doc list.
    Recalculates token counts so all downstream metrics are accurate.

    Senior note: We never mutate the original — we return a new object.
    This makes the optimization pipeline composable and testable.
    """
    docs_tokens = sum(counter.count(d.content) for d in new_docs)

    # Update token counts on the cloned docs
    enriched: list[RetrievedDoc] = []
    for doc in new_docs:
        enriched.append(doc.model_copy(
            update={"token_count": counter.count(doc.content)}
        ))

    return original.model_copy(update={
        "retrieved_docs": enriched,
        "docs_tokens": docs_tokens,
        "total_tokens": (
            original.system_tokens
            + original.user_tokens
            + original.history_tokens
            + docs_tokens
        ),
    })


# ── Strategy 1: Deduplicate ──────────────────────────────────────────────

def strategy_deduplicate(
    entry: ContextEntry,
    cfg: OptimizationConfig,
    counter: TokenCounter,
) -> tuple[ContextEntry, ChangeRecord | None]:
    """
    Remove near-duplicate documents using cosine similarity.

    Algorithm: greedy — iterate docs in order, drop any doc whose
    similarity to ANY already-kept doc exceeds the threshold.
    Keeps the FIRST occurrence (highest retriever score after re-ranking).

    Complexity: O(n²) pairwise — acceptable for n < 50 docs.
    For production RAG with hundreds of docs, use approximate NN instead.
    """
    docs = entry.retrieved_docs
    if len(docs) <= 1:
        return entry, None

    tokens_before = entry.docs_tokens
    vectors = [_tf_vector(doc.content) for doc in docs]
    kept_indices: list[int] = []
    removed_count = 0

    for i, vec in enumerate(vectors):
        is_duplicate = False
        for kept_idx in kept_indices:
            sim = _cosine_sim(vec, vectors[kept_idx])
            if sim > cfg.similarity_threshold:
                is_duplicate = True
                removed_count += 1
                logger.debug(
                    "Dedup: doc[{}] (src={}) similarity={:.2f} to doc[{}] — removed",
                    i, docs[i].source, sim, kept_idx,
                )
                break
        if not is_duplicate:
            kept_indices.append(i)

    if removed_count == 0:
        return entry, None

    kept_docs = [docs[i] for i in kept_indices]
    new_entry = _rebuild_entry(entry, kept_docs, counter)
    tokens_after = new_entry.docs_tokens

    record = ChangeRecord(
        strategy="deduplicate",
        description=(
            f"Removed {removed_count} near-duplicate doc(s) "
            f"(similarity threshold={cfg.similarity_threshold}). "
            f"Kept {len(kept_docs)}/{len(docs)} docs."
        ),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )
    return new_entry, record


# ── Strategy 2: Re-rank + top-K ─────────────────────────────────────────

def strategy_rerank(
    entry: ContextEntry,
    cfg: OptimizationConfig,
    counter: TokenCounter,
) -> tuple[ContextEntry, ChangeRecord | None]:
    """
    Sort docs by relevance and keep the top-K most relevant.

    Ranking signal (in priority order):
      1. retriever relevance_score (if provided by the retrieval system)
      2. term overlap with user_prompt (fallback)

    This means: if your retriever gives you scores, use them.
    If not, we compute our own from keyword overlap.

    Production note: In a real RAG system, retriever scores come from
    embedding cosine distance. Our keyword fallback is a good approximation
    for keyword-dense technical content.
    """
    docs = entry.retrieved_docs
    if len(docs) <= 1:
        return entry, None

    tokens_before = entry.docs_tokens
    prompt_vec = _tf_vector(entry.user_prompt)

    def relevance_score(doc: RetrievedDoc) -> float:
        if doc.relevance_score is not None:
            # blend: 70% retriever score, 30% term overlap
            overlap = _cosine_sim(_tf_vector(doc.content), prompt_vec)
            return 0.70 * doc.relevance_score + 0.30 * overlap
        # pure term overlap fallback
        return _cosine_sim(_tf_vector(doc.content), prompt_vec)

    # Sort descending by relevance
    scored = sorted(docs, key=relevance_score, reverse=True)

    # Apply min_relevance_score filter
    if cfg.min_relevance_score > 0:
        scored = [
            d for d in scored
            if (d.relevance_score or 0.0) >= cfg.min_relevance_score
        ]

    # Keep top-K
    kept = scored[: cfg.max_docs]

    if len(kept) == len(docs) and kept == docs:
        return entry, None  # already in order, nothing removed

    new_entry = _rebuild_entry(entry, kept, counter)
    removed = len(docs) - len(kept)

    record = ChangeRecord(
        strategy="rerank",
        description=(
            f"Re-ranked {len(docs)} docs by relevance score. "
            f"Kept top {len(kept)} (removed {removed} lowest-scoring)."
        ),
        tokens_before=tokens_before,
        tokens_after=new_entry.docs_tokens,
    )
    return new_entry, record


# ── Strategy 3: Trim (per-doc + budget) ─────────────────────────────────

def strategy_trim(
    entry: ContextEntry,
    cfg: OptimizationConfig,
    counter: TokenCounter,
) -> tuple[ContextEntry, ChangeRecord | None]:
    """
    Trim documents to fit within the token budget.

    Two-pass approach:
      Pass 1: Hard cap — any individual doc exceeding max_doc_tokens
              is truncated to the first max_doc_tokens worth of content.
      Pass 2: Budget cap — if total tokens still exceed
              target_token_budget * max_context_tokens, progressively
              truncate the LONGEST remaining docs first.

    Truncation method: sentence-boundary truncation (not mid-word).
    Appends "[truncated]" marker so the model knows content was cut.
    """
    docs = entry.retrieved_docs
    if not docs:
        return entry, None

    tokens_before = entry.docs_tokens
    target_docs_budget = int(cfg.target_token_budget * entry.max_context_tokens)
    # Subtract non-doc tokens to get the actual docs budget
    non_doc_tokens = (
        entry.system_tokens + entry.user_tokens + entry.history_tokens
    )
    available_for_docs = max(100, target_docs_budget - non_doc_tokens)

    def truncate_to_tokens(text: str, max_tok: int) -> str:
        """Truncate text to max_tok tokens at sentence boundary."""
        if counter.count(text) <= max_tok:
            return text
        # Split by sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: list[str] = []
        current_tokens = 0
        for sent in sentences:
            sent_tokens = counter.count(sent)
            if current_tokens + sent_tokens > max_tok:
                break
            result.append(sent)
            current_tokens += sent_tokens
        truncated = " ".join(result)
        if truncated != text:
            truncated += " [truncated]"
        return truncated if truncated.strip() else text[:200] + " [truncated]"

    # Pass 1: Hard cap per doc
    trimmed_docs: list[RetrievedDoc] = []
    for doc in docs:
        new_content = truncate_to_tokens(doc.content, cfg.max_doc_tokens)
        trimmed_docs.append(doc.model_copy(update={"content": new_content}))

    # Pass 2: Budget cap — trim longest docs until within budget
    iteration = 0
    while True:
        current_tokens = sum(counter.count(d.content) for d in trimmed_docs)
        if current_tokens <= available_for_docs:
            break
        if iteration > 20:  # safety valve
            break
        # Find and trim the longest doc
        longest_idx = max(
            range(len(trimmed_docs)),
            key=lambda i: counter.count(trimmed_docs[i].content),
        )
        doc = trimmed_docs[longest_idx]
        current_doc_tokens = counter.count(doc.content)
        # Reduce by 15% each pass
        new_limit = max(50, int(current_doc_tokens * 0.85))
        new_content = truncate_to_tokens(doc.content, new_limit)
        trimmed_docs[longest_idx] = doc.model_copy(
            update={"content": new_content}
        )
        iteration += 1

    new_entry = _rebuild_entry(entry, trimmed_docs, counter)
    tokens_after = new_entry.docs_tokens

    if tokens_after >= tokens_before:
        return entry, None  # nothing changed

    record = ChangeRecord(
        strategy="trim",
        description=(
            f"Trimmed doc content to fit token budget. "
            f"Max per doc: {cfg.max_doc_tokens} tokens. "
            f"Available for docs: {available_for_docs} tokens."
        ),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )
    return new_entry, record


# ── Strategy 4: Compress ────────────────────────────────────────────────

def strategy_compress(
    entry: ContextEntry,
    cfg: OptimizationConfig,
    counter: TokenCounter,
) -> tuple[ContextEntry, ChangeRecord | None]:
    """
    Sentence-level compression — remove sentences with low keyword overlap
    with the user prompt.

    How it works:
      1. Extract keywords from user_prompt
      2. For each doc, score every sentence by keyword overlap
      3. Keep sentences above the keyword_ratio threshold
      4. Never compress below min_sentences_per_doc

    This is a lightweight extractive summarization.
    Production systems use fine-tuned models (e.g. LongT5) for this.
    Our keyword approach works well for technical/factual content.

    Research note: This is related to query-focused extractive summarization,
    a well-studied NLP task. Cite "Query-Focused Summarization" in your README.
    """
    docs = entry.retrieved_docs
    if not docs or not cfg.enable_compression:
        return entry, None

    tokens_before = entry.docs_tokens

    # Extract prompt keywords (4+ chars, skip stopwords)
    STOPWORDS = {
        "that", "this", "with", "from", "have", "will", "been", "were",
        "they", "what", "when", "where", "which", "there", "their",
        "about", "would", "could", "should", "more", "into", "also",
    }
    prompt_keywords = set(
        t for t in re.findall(r"\b[a-z]{4,}\b", entry.user_prompt.lower())
        if t not in STOPWORDS
    )

    if not prompt_keywords:
        return entry, None

    def score_sentence(sent: str) -> float:
        """Fraction of prompt keywords present in this sentence."""
        sent_words = set(re.findall(r"\b[a-z]{4,}\b", sent.lower()))
        if not sent_words:
            return 0.0
        overlap = len(prompt_keywords & sent_words)
        return overlap / len(prompt_keywords)

    compressed_docs: list[RetrievedDoc] = []
    any_changed = False

    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.content.strip())
        if len(sentences) <= cfg.min_sentences_per_doc:
            compressed_docs.append(doc)
            continue

        # Score each sentence
        scored_sents = [(s, score_sentence(s)) for s in sentences if s.strip()]

        # Keep sentences above threshold OR the top min_sentences regardless
        kept = [s for s, score in scored_sents
                if score >= cfg.compression_keyword_ratio]

        # Guarantee minimum sentences
        if len(kept) < cfg.min_sentences_per_doc:
            # Fall back to top-N by score
            sorted_sents = sorted(scored_sents, key=lambda x: x[1], reverse=True)
            kept = [s for s, _ in sorted_sents[: cfg.min_sentences_per_doc]]
            # Restore original order
            original_order = {s: i for i, (s, _) in enumerate(scored_sents)}
            kept = sorted(kept, key=lambda s: original_order.get(s, 999))

        new_content = " ".join(kept)
        if new_content != doc.content and new_content.strip():
            compressed_docs.append(doc.model_copy(update={"content": new_content}))
            any_changed = True
        else:
            compressed_docs.append(doc)

    if not any_changed:
        return entry, None

    new_entry = _rebuild_entry(entry, compressed_docs, counter)
    tokens_after = new_entry.docs_tokens

    record = ChangeRecord(
        strategy="compress",
        description=(
            f"Sentence-level compression using {len(prompt_keywords)} "
            f"prompt keywords (ratio threshold={cfg.compression_keyword_ratio}). "
            f"Removed low-relevance sentences from docs."
        ),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )
    return new_entry, record


# ---------------------------------------------------------------------------
# Main Optimizer
# ---------------------------------------------------------------------------

class ContextOptimizer:
    """
    Reads an EvaluationReport and applies targeted strategies
    to produce a leaner, higher-quality ContextEntry.

    Usage:
        optimizer = ContextOptimizer()
        result = optimizer.optimize(context_entry, evaluation_report)

        print(result.summary())
        print(f"Saved {result.tokens_saved} tokens")

        # Use the improved context for re-run
        improved_entry = result.optimized_entry
    """

    # Map evaluator dimensions to optimizer strategies.
    # Key = dimension name, value = strategy to trigger if score < threshold
    DIMENSION_STRATEGY_MAP = {
        "redundancy":   ("deduplicate", 80.0),
        "relevance":    ("rerank",      75.0),
        "length":       ("trim",        70.0),
        "specificity":  (None,          0.0),   # can't auto-fix vague prompts
        "completeness": (None,          0.0),   # can't auto-add system prompts
    }

    def __init__(self, cfg: OptimizationConfig | None = None):
        self.cfg = cfg or OptimizationConfig()
        self._counter = TokenCounter()

    def optimize(
        self,
        entry: ContextEntry,
        report: EvaluationReport,
    ) -> OptimizationResult:
        """
        Run the full optimization pipeline.

        Strategy selection is driven by the EvaluationReport:
        only strategies that address actual problems are applied.
        This avoids over-optimizing contexts that are already healthy.

        Pipeline order is important:
          1. Deduplicate first — removes redundant docs before scoring
          2. Re-rank — scores the deduplicated set accurately
          3. Trim — fits the re-ranked set into the budget
          4. Compress — fine-grained sentence filtering after trimming

        Args:
            entry:  The original ContextEntry from the logger.
            report: The EvaluationReport from the evaluator.

        Returns:
            OptimizationResult with optimized entry + change log.
        """
        if report.overall_score >= 90:
            logger.info(
                "Optimization skipped — context score {}/100 already excellent.",
                report.overall_score,
            )
            return OptimizationResult(
                original_entry=entry,
                optimized_entry=entry,
                was_modified=False,
            )

        # Determine which strategies to apply based on dimension scores
        dim_scores = {ds.name: ds.score for ds in report.dimension_scores}
        strategies_needed: list[str] = []

        for dim, (strategy, threshold) in self.DIMENSION_STRATEGY_MAP.items():
            if strategy and dim_scores.get(dim, 100.0) < threshold:
                if strategy not in strategies_needed:
                    strategies_needed.append(strategy)

        # Always attempt trim if token utilization > 80%
        if entry.token_utilization > 0.80 and "trim" not in strategies_needed:
            strategies_needed.append("trim")

        # Apply compression after trim when docs are still large
        if entry.docs_tokens > 500 and "compress" not in strategies_needed:
            strategies_needed.append("compress")

        if not strategies_needed:
            logger.info(
                "No optimization strategies needed for run_id={}",
                entry.run_id[:8],
            )
            return OptimizationResult(
                original_entry=entry,
                optimized_entry=entry,
                was_modified=False,
            )

        logger.info(
            "Optimizing run_id={} | strategies={} | score={}/100",
            entry.run_id[:8],
            strategies_needed,
            report.overall_score,
        )

        # Pipeline execution
        current_entry = entry
        change_log: list[ChangeRecord] = []
        applied: list[str] = []

        strategy_fns = {
            "deduplicate": strategy_deduplicate,
            "rerank":      strategy_rerank,
            "trim":        strategy_trim,
            "compress":    strategy_compress,
        }

        # Execute in fixed pipeline order
        for strategy_name in ["deduplicate", "rerank", "trim", "compress"]:
            if strategy_name not in strategies_needed:
                continue
            fn = strategy_fns[strategy_name]
            try:
                new_entry, record = fn(current_entry, self.cfg, self._counter)
                if record is not None:
                    change_log.append(record)
                    applied.append(strategy_name)
                    current_entry = new_entry
                    logger.debug(
                        "  [{}] {} tokens → {} tokens (-{})",
                        strategy_name,
                        record.tokens_before,
                        record.tokens_after,
                        record.tokens_saved,
                    )
            except Exception as e:
                # Strategy failure must never crash optimization
                logger.error(
                    "Strategy '{}' failed for run_id={}: {}",
                    strategy_name, entry.run_id[:8], e,
                )

        was_modified = len(change_log) > 0
        result = OptimizationResult(
            original_entry=entry,
            optimized_entry=current_entry,
            change_log=change_log,
            strategies_applied=applied,
            was_modified=was_modified,
        )

        if was_modified:
            logger.info(
                "Optimization complete | run_id={} | saved={} tokens ({:.1f}%) "
                "| {} docs removed | strategies={}",
                entry.run_id[:8],
                result.tokens_saved,
                result.token_reduction_pct,
                result.docs_removed,
                applied,
            )

        return result

    def optimize_and_report(
        self,
        entry: ContextEntry,
        report: EvaluationReport,
    ) -> tuple[OptimizationResult, EvaluationReport]:
        """
        Optimize context and immediately re-evaluate the result.

        Useful for verifying that optimization actually improved the score
        and for showing before/after comparison in the CLI and dashboard.

        Returns:
            (OptimizationResult, new EvaluationReport on optimized entry)
        """
        from contextlens.core.evaluator import ContextEvaluator

        opt_result = self.optimize(entry, report)

        if not opt_result.was_modified:
            return opt_result, report

        evaluator = ContextEvaluator()
        new_report = evaluator.evaluate(opt_result.optimized_entry)

        logger.info(
            "Score after optimization: {}/100 (was {}/100) | grade: {} → {}",
            new_report.overall_score,
            report.overall_score,
            report.grade,
            new_report.grade,
        )

        return opt_result, new_report