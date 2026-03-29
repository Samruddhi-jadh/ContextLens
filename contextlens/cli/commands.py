# contextlens/cli/commands.py
"""
ContextLens CLI — Phase 7

A production-grade, pip-installable command-line interface built
with Typer. Every command is a thin wrapper that delegates to
the core pipeline (Phases 2–6).

Exit codes (Unix standard — enables CI/CD gates):
    0 = success / context passed evaluation
    1 = context evaluation failed (score < threshold)
    2 = tool error (bad args, missing API key, provider failure)

Usage after `pip install contextlens`:
    contextlens run    "Explain transformers"
    contextlens compare "Explain transformers" --providers groq gemini
    contextlens analyze logs/default_20241201.jsonl
    contextlens report  --session research_agent --export
    contextlens check
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from loguru import logger

app = typer.Typer(
    name="contextlens",
    help=(
        "ContextLens — Context Engineering & Evaluation Toolkit for AI Agents.\n\n"
        "Monitor, evaluate, and optimize context passed to AI agents across "
        "Groq, Gemini, and NVIDIA providers."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Shared enums + helpers
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    groq   = "groq"
    gemini = "gemini"
    nvidia = "nvidia"


def _exit_error(msg: str, code: int = 2) -> None:
    err_console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code=code)


def _print_version(value: bool) -> None:
    if value:
        from contextlens import __version__
        console.print(f"contextlens v{__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Version flag
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v",
        callback=_print_version,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


# ---------------------------------------------------------------------------
# contextlens run
# ---------------------------------------------------------------------------

@app.command()
def run(
    prompt: str = typer.Argument(..., help="The prompt to send to the AI agent."),
    provider: Provider = typer.Option(
        Provider.groq, "--provider", "-p",
        help="AI provider to use.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model override (e.g. llama-3.1-70b-versatile).",
    ),
    system: str = typer.Option(
        "You are a helpful and precise AI assistant.",
        "--system", "-s",
        help="System prompt.",
    ),
    max_tokens: int = typer.Option(
        1024, "--max-tokens",
        help="Maximum response tokens.",
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t",
        help="Sampling temperature (0.0 = deterministic).",
    ),
    session: str = typer.Option(
        "cli", "--session",
        help="Session ID for grouping runs in logs.",
    ),
    budget: Optional[float] = typer.Option(
        None, "--budget",
        help="Cost budget in USD. Warns at 80%, errors at 100%.",
    ),
    export: Optional[Path] = typer.Option(
        None, "--export", "-e",
        help="Export full report as JSON to this path.",
    ),
    optimize: bool = typer.Option(
        False, "--optimize", "-o",
        help="Run context optimizer before inference.",
    ),
    pass_threshold: float = typer.Option(
        70.0, "--threshold",
        help="Minimum evaluation score to exit 0. Below = exit 1.",
    ),
) -> None:
    """
    Run a single prompt through the full ContextLens pipeline.

    Logs context, evaluates quality, runs inference, records metrics.
    Prints a rich summary table to the terminal.

    [bold]Examples:[/bold]

        contextlens run "Explain attention mechanisms in transformers"

        contextlens run "Write a Python async HTTP client" \\
            --provider groq --model llama-3.1-70b-versatile \\
            --system "You are a senior Python engineer." \\
            --optimize --export report.json
    """
    try:
        from contextlens import ContextLens

        lens = ContextLens(
            session_id=session,
            default_provider=provider.value,
            budget_usd=budget,
        )

        console.print(f"\n[dim]Running on {provider.value}...[/dim]")

        result = lens.run(
            prompt=prompt,
            provider=provider.value,
            model=model,
            system_prompt=system,
            max_tokens=max_tokens,
            temperature=temperature,
            tags=["cli", "run"],
        )

        # Optionally optimize and re-run
        if optimize and result.evaluation and result.evaluation.overall_score < pass_threshold:
            console.print("[yellow]Context below threshold — running optimizer...[/yellow]")
            opt_result = lens.optimize(result)
            if opt_result.was_modified:
                console.print(
                    f"[green]Optimized: saved {opt_result.tokens_saved} tokens "
                    f"({opt_result.token_reduction_pct:.1f}% reduction)[/green]"
                )
                result = lens.run_entry(
                    opt_result.optimized_entry,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        result.print_summary()

        if export:
            lens.export_report(export)
            console.print(f"[dim]Report exported → {export}[/dim]")

        # Exit code based on evaluation score
        score = result.evaluation.overall_score if result.evaluation else 100.0
        if score < pass_threshold:
            raise typer.Exit(code=1)

    except (ImportError, EnvironmentError, ValueError) as e:
        _exit_error(str(e))


# ---------------------------------------------------------------------------
# contextlens compare
# ---------------------------------------------------------------------------

@app.command()
def compare(
    prompt: str = typer.Argument(..., help="Prompt to compare across providers."),
    providers: list[Provider] = typer.Option(
        [Provider.groq, Provider.gemini],
        "--providers", "-p",
        help="Providers to compare (repeat flag for multiple).",
    ),
    system: str = typer.Option(
        "You are a helpful and precise AI assistant.",
        "--system", "-s",
        help="System prompt.",
    ),
    max_tokens: int = typer.Option(
        512, "--max-tokens",
        help="Max tokens per response.",
    ),
    session: str = typer.Option(
        "cli_compare", "--session",
        help="Session ID.",
    ),
    export: Optional[Path] = typer.Option(
        None, "--export", "-e",
        help="Export comparison report as JSON.",
    ),
) -> None:
    """
    Run the same prompt across multiple providers and compare results.

    Ranks by latency, shows cost and token breakdown per provider.

    [bold]Examples:[/bold]

        contextlens compare "Explain gradient descent in 3 sentences" \\
            --providers groq --providers gemini

        contextlens compare "Write a quicksort in Python" \\
            --providers groq --providers gemini --providers nvidia \\
            --export comparison.json
    """
    try:
        from contextlens import ContextLens

        lens = ContextLens(session_id=session)
        provider_names = [p.value for p in providers]

        console.print(
            f"\n[dim]Comparing across: {', '.join(provider_names)}...[/dim]\n"
        )

        results = lens.compare(
            prompt=prompt,
            providers=provider_names,
            system_prompt=system,
            max_tokens=max_tokens,
        )

        if not results:
            _exit_error("No results returned — check your API keys with: contextlens check")

        # Detailed per-provider panel
        for r in results:
            score_str = ""
            if r.evaluation:
                color = "green" if r.evaluation.overall_score >= 70 else "yellow"
                score_str = f" | Context: [{color}]{r.evaluation.overall_score:.0f}/100[/{color}]"
            console.print(
                f"[bold]{r.response.provider}[/bold]  "
                f"{r.response.latency_ms:.0f}ms  "
                f"${r.response.estimated_cost_usd:.6f}"
                f"{score_str}"
            )
            console.print(f"[dim]{r.response.text[:120].replace(chr(10), ' ')}...[/dim]\n")

        if export:
            export_data = [r.response.to_dict() for r in results]
            export.write_text(json.dumps(export_data, indent=2))
            console.print(f"[dim]Comparison exported → {export}[/dim]")

    except (ImportError, EnvironmentError, ValueError) as e:
        _exit_error(str(e))


# ---------------------------------------------------------------------------
# contextlens analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    log_file: Path = typer.Argument(
        ...,
        help="Path to a JSONL context log file (from logs/ directory).",
        exists=True,
    ),
    limit: int = typer.Option(
        20, "--limit", "-n",
        help="Number of most recent entries to analyze.",
    ),
    threshold: float = typer.Option(
        70.0, "--threshold",
        help="Score threshold — entries below this are flagged.",
    ),
    export: Optional[Path] = typer.Option(
        None, "--export", "-e",
        help="Export analysis results as JSON.",
    ),
    show_suggestions: bool = typer.Option(
        True, "--suggestions/--no-suggestions",
        help="Show optimization suggestions.",
    ),
) -> None:
    """
    Analyze a context log file and evaluate every entry offline.

    Reads a JSONL log produced by ContextLens, re-runs the evaluator
    on each entry, and prints a scored summary table.

    [bold]Examples:[/bold]

        contextlens analyze logs/default_20241201.jsonl

        contextlens analyze logs/research_agent_20241201.jsonl \\
            --limit 50 --threshold 75 --export analysis.json
    """
    try:
        import json as _json
        from contextlens.core.logger import ContextEntry, RetrievedDoc
        from contextlens.core.evaluator import ContextEvaluator

        evaluator = ContextEvaluator()
        entries_raw: list[dict] = []

        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries_raw.append(_json.loads(line))
                    except _json.JSONDecodeError:
                        continue

        if not entries_raw:
            _exit_error(f"No valid entries found in {log_file}")

        # Take most recent N
        entries_raw = entries_raw[-limit:]
        console.print(
            f"\n[dim]Analyzing {len(entries_raw)} entries from {log_file.name}...[/dim]\n"
        )

        table = Table(
            title=f"Context Analysis — {log_file.name}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Run ID",     width=14)
        table.add_column("Provider",   width=10)
        table.add_column("Tokens",     width=8)
        table.add_column("Score",      width=8)
        table.add_column("Grade",      width=6)
        table.add_column("Top Issue",  width=46)

        results_export = []
        failed_count = 0

        for raw in entries_raw:
            # Reconstruct ContextEntry from stored dict
            raw_docs = raw.get("retrieved_docs", [])
            docs = [
                RetrievedDoc(
                    content=d.get("content", ""),
                    source=d.get("source", "unknown"),
                    relevance_score=d.get("relevance_score"),
                    token_count=d.get("token_count", 0),
                )
                for d in raw_docs
            ]

            entry = ContextEntry(
                run_id=raw.get("run_id", ""),
                session_id=raw.get("session_id", ""),
                user_prompt=raw.get("user_prompt", ""),
                system_prompt=raw.get("system_prompt", ""),
                retrieved_docs=docs,
                conversation_history=raw.get("conversation_history", []),
                system_tokens=raw.get("system_tokens", 0),
                user_tokens=raw.get("user_tokens", 0),
                docs_tokens=raw.get("docs_tokens", 0),
                history_tokens=raw.get("history_tokens", 0),
                total_tokens=raw.get("total_tokens", 0),
                provider=raw.get("provider", "unknown"),
                model=raw.get("model", "unknown"),
                max_context_tokens=raw.get("max_context_tokens", 8000),
                tags=raw.get("tags", []),
            )

            report = evaluator.evaluate(entry)
            top_issue = report.all_issues[0][:44] if report.all_issues else "—"
            score_color = (
                "green" if report.overall_score >= 80
                else "yellow" if report.overall_score >= threshold
                else "red"
            )

            table.add_row(
                entry.run_id[:12] + "...",
                entry.provider,
                str(entry.total_tokens),
                f"[{score_color}]{report.overall_score:.1f}[/{score_color}]",
                f"[{score_color}]{report.grade}[/{score_color}]",
                top_issue,
            )

            if report.overall_score < threshold:
                failed_count += 1

            results_export.append({
                "run_id": entry.run_id,
                "score": report.overall_score,
                "grade": report.grade,
                "issues": report.all_issues,
                "suggestions": report.all_suggestions if show_suggestions else [],
            })

        console.print(table)

        # Summary footer
        pass_count = len(entries_raw) - failed_count
        footer_color = "green" if failed_count == 0 else "yellow"
        console.print(
            f"\n[{footer_color}]"
            f"Passed: {pass_count}/{len(entries_raw)}  |  "
            f"Failed: {failed_count}/{len(entries_raw)}  |  "
            f"Threshold: {threshold}/100"
            f"[/{footer_color}]"
        )

        if show_suggestions and any(r["suggestions"] for r in results_export):
            console.print("\n[bold]Top suggestions across session:[/bold]")
            seen: set[str] = set()
            for r in results_export:
                for s in r["suggestions"]:
                    if s not in seen:
                        console.print(f"  [cyan]→[/cyan] {s[:90]}")
                        seen.add(s)
                    if len(seen) >= 5:
                        break

        if export:
            export.write_text(json.dumps(results_export, indent=2))
            console.print(f"\n[dim]Analysis exported → {export}[/dim]")

        # Exit 1 if any entries failed
        if failed_count > 0:
            raise typer.Exit(code=1)

    except (FileNotFoundError, PermissionError) as e:
        _exit_error(str(e))


# ---------------------------------------------------------------------------
# contextlens report
# ---------------------------------------------------------------------------

@app.command()
def report(
    session: str = typer.Option(
        "cli", "--session", "-s",
        help="Session ID to report on.",
    ),
    log_dir: Path = typer.Option(
        Path("./logs"), "--log-dir",
        help="Log directory to read metrics from.",
    ),
    budget: Optional[float] = typer.Option(
        None, "--budget",
        help="Budget in USD for budget utilization display.",
    ),
    export: Optional[Path] = typer.Option(
        None, "--export", "-e",
        help="Export report as JSON to this path.",
    ),
) -> None:
    """
    Print a full session monitoring report: tokens, cost, latency.

    Reads metrics from the JSONL metrics store and generates a
    rich terminal report with latency percentiles and provider breakdown.

    [bold]Examples:[/bold]

        contextlens report

        contextlens report --session research_agent \\
            --budget 0.50 --export session_report.json
    """
    try:
        from contextlens.core.monitor import TokenMonitor

        monitor = TokenMonitor(
            session_id=session,
            budget_usd=budget,
            log_dir=log_dir,
        )

        if monitor.run_count == 0:
            console.print(
                Panel(
                    f"[yellow]No runs found for session '[bold]{session}[/bold]' "
                    f"in {log_dir}[/yellow]\n\n"
                    "Run [bold]contextlens run[/bold] first to generate data.",
                    title="No data",
                    border_style="yellow",
                )
            )
            raise typer.Exit(code=0)

        monitor.print_report()

        if export:
            out = monitor.export_report_json(export)
            console.print(f"[dim]Report exported → {out}[/dim]")

    except (FileNotFoundError, PermissionError) as e:
        _exit_error(str(e))


# ---------------------------------------------------------------------------
# contextlens check
# ---------------------------------------------------------------------------

@app.command()
def check(
    providers: list[Provider] = typer.Option(
        list(Provider),
        "--providers", "-p",
        help="Providers to check (default: all configured).",
    ),
) -> None:
    """
    Health-check all configured AI providers.

    Sends a minimal test message to each provider and reports
    whether the connection and API key are working.

    [bold]Examples:[/bold]

        contextlens check

        contextlens check --providers groq --providers gemini
    """
    from contextlens.providers import PROVIDER_REGISTRY
    from contextlens.config import config

    table = Table(
        title="Provider Health Check",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Provider",   width=12)
    table.add_column("API Key",    width=14)
    table.add_column("Status",     width=10)
    table.add_column("Latency",    width=10)
    table.add_column("Note",       width=36)

    all_ok = True

    for provider in providers:
        name = provider.value
        key_attr = f"{name}_api_key"
        has_key = bool(getattr(config, key_attr, ""))
        key_display = "[green]set[/green]" if has_key else "[red]missing[/red]"

        if not has_key:
            table.add_row(
                name, key_display,
                "[red]skip[/red]", "—",
                f"Set {name.upper()}_API_KEY in .env",
            )
            all_ok = False
            continue

        try:
            import time
            provider_cls = PROVIDER_REGISTRY[name]
            p = provider_cls()
            start = time.perf_counter()
            ok = p.health_check()
            latency = (time.perf_counter() - start) * 1000

            if ok:
                table.add_row(
                    name, key_display,
                    "[green]pass[/green]",
                    f"{latency:.0f} ms",
                    "Connection OK",
                )
            else:
                table.add_row(
                    name, key_display,
                    "[red]fail[/red]",
                    f"{latency:.0f} ms",
                    "No response from provider",
                )
                all_ok = False

        except Exception as e:
            table.add_row(
                name, key_display,
                "[red]error[/red]", "—",
                str(e)[:36],
            )
            all_ok = False

    console.print(table)

    if all_ok:
        console.print("\n[green]All providers healthy.[/green]")
    else:
        console.print(
            "\n[yellow]Some providers failed. "
            "Check .env and run: contextlens check[/yellow]"
        )
        raise typer.Exit(code=2)