# contextlens/lens.py
"""
ContextLens — main orchestrator.

This is THE public API. Everything else is an implementation detail.

Usage:
    from contextlens import ContextLens

    lens = ContextLens()
    result = lens.run(
        prompt="Explain attention mechanisms",
        provider="groq",
    )
    print(result.response.text)
    print(result.context_entry.total_tokens)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from contextlens.config import config
from contextlens.core.logger import ContextEntry, ContextLogger, RetrievedDoc
from contextlens.core.evaluator import ContextEvaluator, EvaluationReport
from contextlens.core.monitor import TokenMonitor, RunMetric
from contextlens.core.optimizer import ContextOptimizer, OptimizationResult
from contextlens.providers import PROVIDER_REGISTRY, ModelResponse, ProviderError

console = Console()


# ---------------------------------------------------------------------------
# LensResult — single definition, complete
# ---------------------------------------------------------------------------

@dataclass
class LensResult:
    """
    The complete output of a lens.run() call.

    Bundles context + response + timing + evaluation into one object.
    This flows into the evaluator (Phase 4), monitor (Phase 5),
    and optimizer (Phase 6).
    """
    context_entry: ContextEntry         # what was logged (Phase 2)
    response: ModelResponse             # what the model said (Phase 3)
    total_wall_time_ms: float = 0.0    # end-to-end time including logging
    evaluation: EvaluationReport | None = None  # quality score (Phase 4)

    def print_summary(self) -> None:
        """Rich-formatted summary table — used by CLI and demos."""
        table = Table(
            title=f"ContextLens Run — {self.response.provider} / {self.response.model}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Metric", style="dim", width=24)
        table.add_column("Value", width=36)

        table.add_row("Run ID",        self.context_entry.run_id[:16] + "...")
        table.add_row("Provider",      self.response.provider)
        table.add_row("Model",         self.response.model)
        table.add_row("Input tokens",  str(self.context_entry.total_tokens))
        table.add_row("Output tokens", str(self.response.output_tokens))
        table.add_row("Total tokens",  str(self.response.total_tokens))
        table.add_row("Context util.", f"{self.context_entry.token_utilization:.1%}")
        table.add_row("API latency",   f"{self.response.latency_ms:.0f} ms")
        table.add_row("Wall time",     f"{self.total_wall_time_ms:.0f} ms")
        table.add_row("Est. cost",     f"${self.response.estimated_cost_usd:.6f} USD")
        console.print(table)
        console.print(f"\n[bold]Response:[/bold]\n{self.response.text}\n")

        # Phase 4 evaluation panel
        if self.evaluation:
            score_color = (
                "green"  if self.evaluation.overall_score >= 80
                else "yellow" if self.evaluation.overall_score >= 60
                else "red"
            )
            lines = [
                f"[{score_color}]Overall: {self.evaluation.overall_score}/100  "
                f"Grade: {self.evaluation.grade}[/{score_color}]"
            ]
            for ds in self.evaluation.dimension_scores:
                bar_filled = int(ds.score / 5)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                lines.append(
                    f"  {ds.name:<14} [{bar}] {ds.score:.0f}  ({ds.grade_letter})"
                )
            if self.evaluation.all_issues:
                lines.append("\n[yellow]Issues:[/yellow]")
                for issue in self.evaluation.all_issues[:4]:
                    lines.append(f"  • {issue[:80]}")
            if self.evaluation.all_suggestions:
                lines.append("\n[cyan]Suggestions:[/cyan]")
                for sug in self.evaluation.all_suggestions[:3]:
                    lines.append(f"  → {sug[:80]}")
            console.print(Panel(
                "\n".join(lines),
                title="Context Quality Evaluation",
                border_style="dim",
            ))


# ---------------------------------------------------------------------------
# ContextLens — single definition, complete
# ---------------------------------------------------------------------------

class ContextLens:
    """
    Main entry point for the ContextLens toolkit.

    Wires together:
      - ContextLogger  (Phase 2) — captures and persists context
      - ModelRouter    (Phase 3) — routes to correct AI provider
      - ContextEvaluator (Phase 4) — scores context quality
      - TokenMonitor   (Phase 5) — tracks cost and latency
      - ContextOptimizer (Phase 6) — improves context automatically

    Args:
        session_id:       Groups related runs in the same log file.
        default_provider: Which provider to use if not specified per-run.
        budget_usd:       Optional cost budget. Warns at 80%, errors at 100%.
    """

    def __init__(
        self,
        session_id: str = "default",
        default_provider: str | None = None,
        budget_usd: float | None = None,
    ):
        self.session_id = session_id
        self.default_provider = default_provider or config.default_provider
        self._context_logger = ContextLogger(session_id=session_id)
        self._provider_cache: dict[str, Any] = {}
        self._evaluator = ContextEvaluator()
        self._monitor = TokenMonitor(
            session_id=session_id,
            budget_usd=budget_usd,
        )
        self._optimizer = ContextOptimizer()

        logger.info(
            "ContextLens initialized | session={} | default_provider={} | budget={}",
            session_id,
            self.default_provider,
            f"${budget_usd:.4f}" if budget_usd else "none",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_provider(self, provider_name: str):
        """
        Lazy-initialize providers — only create SDK clients when needed.
        Caches them so we reuse connections within a session.
        """
        if provider_name not in self._provider_cache:
            if provider_name not in PROVIDER_REGISTRY:
                raise ValueError(
                    f"Unknown provider '{provider_name}'. "
                    f"Choose from: {list(PROVIDER_REGISTRY)}"
                )
            config.validate_provider(provider_name)
            self._provider_cache[provider_name] = PROVIDER_REGISTRY[provider_name]()
            logger.debug("Provider '{}' initialized and cached.", provider_name)
        return self._provider_cache[provider_name]

    def _build_messages(self, entry: ContextEntry) -> list[dict[str, str]]:
        """
        Convert a ContextEntry into the OpenAI-format messages list
        that every provider adapter expects.

        Message ordering matters for LLM performance:
          1. system  — sets persona and constraints
          2. history — prior conversation turns
          3. docs    — retrieved context injected as user turn
          4. prompt  — the actual user question
        """
        messages: list[dict[str, str]] = []

        if entry.system_prompt:
            messages.append({"role": "system", "content": entry.system_prompt})

        messages.extend(entry.conversation_history)

        if entry.retrieved_docs:
            docs_text = "\n\n".join(
                f"[Source: {doc.source}]\n{doc.content}"
                for doc in entry.retrieved_docs
            )
            messages.append({
                "role": "user",
                "content": (
                    "Use the following retrieved context to answer the question:\n\n"
                    + docs_text
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I'll use this context to answer.",
            })

        messages.append({"role": "user", "content": entry.user_prompt})
        return messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        system_prompt: str = "",
        retrieved_docs: list[RetrievedDoc] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LensResult:
        """
        Run a complete context-logged AI inference.

        Logs context → evaluates quality → calls provider → records metrics.
        Returns a LensResult with full observability data attached.
        """
        wall_start = time.perf_counter()
        provider_name = provider or self.default_provider
        active_provider = self._get_provider(provider_name)
        model_name = model or active_provider.default_model

        # Phase 2: log the context
        context_entry = self._context_logger.log(
            user_prompt=prompt,
            system_prompt=system_prompt,
            retrieved_docs=retrieved_docs,
            conversation_history=conversation_history,
            provider=provider_name,
            model=model_name,
            tags=tags,
            metadata=metadata,
        )

        # Phase 3: build messages and call provider
        messages = self._build_messages(context_entry)
        try:
            response = active_provider.complete(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response.run_id = context_entry.run_id
        except ProviderError as e:
            logger.error("Provider call failed: {}", e)
            raise

        wall_time_ms = (time.perf_counter() - wall_start) * 1000

        # Phase 4: evaluate context quality
        evaluation = self._evaluator.evaluate(context_entry)

        result = LensResult(
            context_entry=context_entry,
            response=response,
            total_wall_time_ms=wall_time_ms,
            evaluation=evaluation,
        )

        # Phase 5: record metrics
        self._monitor.record(result)

        logger.info(
            "Run complete | {} tokens out | {:.0f}ms | ${:.6f}",
            response.output_tokens,
            response.latency_ms,
            response.estimated_cost_usd,
        )

        return result

    def run_entry(
        self,
        entry: ContextEntry,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LensResult:
        """
        Run inference on a pre-built (possibly optimized) ContextEntry.

        Used after optimization to re-run with the improved context
        without re-logging or re-evaluating the original.
        """
        provider_name = provider or entry.provider or self.default_provider
        active_provider = self._get_provider(provider_name)
        model_name = model or entry.model or active_provider.default_model

        wall_start = time.perf_counter()
        messages = self._build_messages(entry)

        try:
            response = active_provider.complete(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response.run_id = entry.run_id
        except Exception as e:
            logger.error("run_entry failed: {}", e)
            raise

        wall_time_ms = (time.perf_counter() - wall_start) * 1000
        evaluation = self._evaluator.evaluate(entry)

        result = LensResult(
            context_entry=entry,
            response=response,
            total_wall_time_ms=wall_time_ms,
            evaluation=evaluation,
        )
        self._monitor.record(result)
        return result

    def compare(
        self,
        prompt: str,
        providers: list[str] | None = None,
        system_prompt: str = "",
        retrieved_docs: list[RetrievedDoc] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> list[LensResult]:
        """
        Run the SAME prompt across multiple providers and return all results.
        Results are sorted by API latency ascending (fastest first).
        """
        target_providers = providers or [
            p for p in PROVIDER_REGISTRY
            if getattr(config, f"{p}_api_key", "")
        ]

        if not target_providers:
            raise ValueError(
                "No providers configured. Set at least one API key in .env"
            )

        results: list[LensResult] = []
        for pname in target_providers:
            try:
                result = self.run(
                    prompt=prompt,
                    provider=pname,
                    system_prompt=system_prompt,
                    retrieved_docs=retrieved_docs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tags=["compare"],
                )
                results.append(result)
            except (ProviderError, ValueError) as e:
                logger.warning("Skipping provider '{}': {}", pname, e)

        results.sort(key=lambda r: r.response.latency_ms)
        self._print_comparison_table(results)
        return results

    def optimize(self, result: LensResult) -> OptimizationResult:
        """
        Optimize the context from a previous run.

        Reads the evaluation from the LensResult and applies targeted
        strategies to produce a leaner ContextEntry.

        Usage:
            result = lens.run(prompt="...", ...)
            opt = lens.optimize(result)
            if opt.was_modified:
                better = lens.run_entry(opt.optimized_entry)
        """
        evaluation = result.evaluation or self._evaluator.evaluate(
            result.context_entry
        )
        return self._optimizer.optimize(result.context_entry, evaluation)

    def report(self) -> None:
        """Print the session monitoring report to the terminal."""
        self._monitor.print_report()

    def export_report(self, output_path: Path | None = None) -> Path:
        """Export the session report as a JSON file."""
        return self._monitor.export_report_json(output_path)

    # ------------------------------------------------------------------
    # Internal display helpers
    # ------------------------------------------------------------------

    def _print_comparison_table(self, results: list[LensResult]) -> None:
        """Rich comparison table across providers."""
        if not results:
            console.print("[yellow]No results to compare.[/yellow]")
            return

        table = Table(
            title="Multi-Model Comparison",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Provider",         width=10)
        table.add_column("Model",            width=28)
        table.add_column("Latency",          width=10)
        table.add_column("Out tokens",       width=10)
        table.add_column("Est. cost",        width=14)
        table.add_column("Response preview", width=34)

        for r in results:
            table.add_row(
                r.response.provider,
                r.response.model,
                f"{r.response.latency_ms:.0f} ms",
                str(r.response.output_tokens),
                f"${r.response.estimated_cost_usd:.6f}",
                r.response.text[:60].replace("\n", " ") + "...",
            )
        console.print(table)
