# ContextLens

> The observability layer your LLM stack is missing.

[![PyPI version](https://badge.fury.io/py/contextlens-ai.svg)](https://pypi.org/project/contextlens-ai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/Samruddhi-jadh/ContextLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Samruddhi-jadh/ContextLens/actions)

**ContextLens** is a production-grade, pip-installable toolkit that makes the context passed to AI agents **visible, measurable, and automatically improvable** — across Groq, Gemini, and NVIDIA providers.

Built as a real-world demonstration of context engineering, multi-model observability, and automated LLM pipeline repair.

---

## The Problem

Every team building LLM-powered products is flying blind. They can see the output but have zero visibility into **why** the context they sent produced that output, what it cost, and how to fix it when it breaks.

| Silent failure | Real consequence |
|---|---|
| Redundant retrieved docs | Wasted tokens, diluted signal, higher cost |
| Irrelevant context injected | Hallucinations, off-topic responses |
| Context window overflow | Silent API truncation, failed requests |
| No cost visibility | Surprise bills, no budget control |
| No quality baseline | No way to know if prompts are good |
| No debugging trail | Cannot reproduce or fix failures |

ContextLens solves all six with a single `pip install`.

---

## Live Demo Output

Real API call output from a production run across 3 providers:

```
Session: research_assistant_demo   Runs: 14   Total tokens: 2,832   Total cost: $0.001431
Budget: $0.001431 of $0.0500 (2.9%) — OK
Avg context quality: 78.2/100 | Passed: 14/14

Provider Breakdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provider   Runs   Total tokens   Total cost      Avg latency   Cost/run
groq       8      1,671          $0.001138        706 ms        $0.000142
gemini     2      438            $0.000014       2403 ms        $0.000007
nvidia     4      723            $0.000278      17578 ms        $0.000069

Optimizer result: 107 tokens saved (54.0% reduction) | grade D → C
```

---

## Architecture

```
Developer prompt
      │
      ▼
┌──────────────────────────────────────────────────────┐
│                     ContextLens                      │
│                                                      │
│   Logger → Evaluator → Optimizer → Monitor          │
│                                                      │
│            Multi-Model Router                        │
│     ┌──────────┬──────────┬──────────┐              │
│     │   Groq   │  Gemini  │  NVIDIA  │              │
│     └──────────┴──────────┴──────────┘              │
└──────────────────────────────────────────────────────┘
      │
      ▼
  LensResult
  ├── response.text
  ├── context_entry  (logged, token-counted, timestamped)
  ├── evaluation     (5-dimension score 0–100, grade A–F)
  └── metrics        (cost USD, latency ms, token breakdown)
```

---

## Installation

```bash
pip install contextlens-ai
```

From source:

```bash
git clone https://github.com/Samruddhi-jadh/ContextLens.git
cd ContextLens
pip install -e ".[dev]"
cp .env.example .env
```

Required API keys in `.env`:

```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxx
GEMINI_API_KEY=AIzaSyxxxxxxxxx
NVIDIA_API_KEY=nvapi_xxxxxxxxx   # optional
```

Get free keys at: [console.groq.com](https://console.groq.com) · [aistudio.google.com](https://aistudio.google.com) · [build.nvidia.com](https://build.nvidia.com)

---

## Quickstart

### Drop-in replacement for direct API calls

```python
# Before — zero visibility
from groq import Groq
client = Groq(api_key="...")
response = client.chat.completions.create(model="...", messages=[...])

# After — full observability, same result
from contextlens import ContextLens

lens = ContextLens(session_id="my_agent", budget_usd=1.00)
result = lens.run(
    prompt="Explain how context engineering reduces hallucinations.",
    system_prompt="You are a senior AI engineer. Be precise.",
)

print(result.response.text)
print(f"Score:   {result.evaluation.overall_score}/100")
print(f"Cost:    ${result.response.estimated_cost_usd:.6f}")
print(f"Latency: {result.response.latency_ms:.0f}ms")
```

### RAG pipeline integration

```python
from contextlens import ContextLens
from contextlens.core.logger import RetrievedDoc

lens = ContextLens(session_id="doc_qa")

# Wrap your existing retriever chunks
docs = [
    RetrievedDoc(
        content=chunk.page_content,
        source=chunk.metadata["source"],
        relevance_score=chunk.score,
    )
    for chunk in vector_store.similarity_search(query, k=8)
]

result = lens.run(
    prompt=user_question,
    system_prompt="You are a document analyst. Answer only from provided documents.",
    retrieved_docs=docs,
    provider="groq",
)
```

### Auto-optimize bad context before re-running

```python
result = lens.run(prompt="...", retrieved_docs=many_docs)

if result.evaluation.overall_score < 70:
    opt = lens.optimize(result)
    print(opt.summary())
    # "Optimized: +107 tokens saved (54.0% reduction)"
    # "Strategies: deduplicate, rerank, trim"
    # "Docs removed: 2"

    better = lens.run_entry(opt.optimized_entry)
    print(f"Score improved: {result.evaluation.overall_score:.1f} → {better.evaluation.overall_score:.1f}")
```

### Multi-provider comparison

```python
results = lens.compare(
    prompt="Explain attention mechanisms in one paragraph.",
    providers=["groq", "gemini", "nvidia"],
)
# Prints rich table sorted by latency:
# groq: 706ms  $0.000142  ✓
# gemini: 2403ms  $0.000007  ✓
# nvidia: 17578ms  $0.000069  ✓
```

---

## CLI

```bash
# Full observability on a single run
contextlens run "Explain transformers" --provider groq --optimize

# Multi-provider benchmark
contextlens compare "Write a Python async HTTP client" \
    --providers groq --providers gemini

# Evaluate a context log file offline
contextlens analyze logs/agent_20250328.jsonl --threshold 70

# Session cost + latency report with budget tracking
contextlens report --session my_agent --budget 1.00 --export report.json

# Health check all configured providers
contextlens check
```

---

## Evaluation Engine

Every run is automatically scored 0–100 across 5 dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Length | 20% | Token budget utilization — ideal range 20–80% |
| Redundancy | 25% | Cosine similarity across retrieved docs |
| Relevance | 25% | Term overlap between docs and prompt keywords |
| Specificity | 15% | Prompt precision, constraints, action clarity |
| Completeness | 15% | System prompt presence, role definition, history |

Grades: **A** (90+) · **B** (80+) · **C** (70+) · **D** (55+) · **F** (<55)

The evaluator produces actionable output:

```
Overall: 76.67/100  Grade: C

  length         [████████████░░░░░░░░] 60  (D)
  redundancy     [███████████████░░░░░] 80  (C)
  relevance      [█████████████░░░░░░░] 68  (D)
  specificity    [██████████████████░░] 93  (A)
  completeness   [██████████████████░░] 92  (A)

Issues:
  • Sparse context: only 1.1% of token budget used
  • System prompt has no behavioural constraints

Suggestions:
  → Retrieve 2–3 more relevant documents
  → Add constraints: 'Always cite sources', 'Respond only in JSON'
```

---

## Optimization Strategies

When evaluation finds problems, the optimizer applies targeted fixes automatically:

| Strategy | Triggered when | What it does |
|---|---|---|
| Deduplicate | Redundancy score < 80 | Removes near-duplicate docs via cosine similarity |
| Re-rank | Relevance score < 75 | Sorts docs by relevance score, keeps top-K |
| Trim | Token utilization > 80% | Truncates long docs to fit token budget |
| Compress | Docs tokens > 500 | Sentence-level filtering by prompt keyword overlap |

Strategies compose in a fixed pipeline — deduplicate first, then rerank, then trim, then compress. Each stage feeds the next. The original context is never mutated.

---

## Token Monitor

Every session produces a full observability report:

```bash
contextlens report --session my_agent --budget 5.00
```

```
Session Overview
─────────────────────────────────────────────────────
Runs: 14   Tokens: 2,832   Cost: $0.001431
Budget: 2.9% used — OK
Quality: 78.2/100 avg | 14/14 passed

Latency Percentiles
─────────────────────
p50 (median)   850 ms
p90           8,893 ms
p99          42,047 ms

Provider Breakdown
─────────────────────────────────────────────────────────────────────
Provider   Runs   Tokens   Cost          Avg latency   Cost/run
groq       8      1,671    $0.001138     706 ms         $0.000142
gemini     2      438      $0.000014    2403 ms         $0.000007
nvidia     4      723      $0.000278   17578 ms         $0.000069
```

---

## CI/CD Integration

ContextLens exits with code `1` when context quality falls below threshold — use it as a deployment gate:

```yaml
# .github/workflows/quality_gate.yml
- name: Context quality gate
  run: |
    contextlens analyze logs/production_$(date +%Y%m%d).jsonl \
      --threshold 70
  # Blocks deployment if any run scores below 70
```

Exit codes follow Unix standard:
- `0` — all runs passed evaluation
- `1` — one or more runs below threshold
- `2` — tool error (missing key, bad file, provider failure)

---

## Project Structure

```
contextlens/
├── contextlens/
│   ├── __init__.py          # Public API surface
│   ├── lens.py              # Main orchestrator — start here
│   ├── config.py            # Pydantic settings, .env loading
│   ├── core/
│   │   ├── logger.py        # Context capture + token counting
│   │   ├── evaluator.py     # 5-dimension scoring engine
│   │   ├── monitor.py       # Token, cost, latency tracking
│   │   └── optimizer.py     # Automated context repair
│   ├── providers/
│   │   ├── base.py          # Abstract provider interface
│   │   ├── groq_provider.py
│   │   ├── gemini_provider.py
│   │   └── nvidia_provider.py
│   └── cli/
│       └── commands.py      # Typer CLI — 5 commands
├── examples/
│   └── research_assistant.py
├── tests/
│   ├── test_logger.py
│   ├── test_evaluator.py
│   ├── test_monitor.py
│   ├── test_optimizer.py
│   ├── test_models.py
│   └── test_cli.py
├── .env.example
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Design Decisions

These are the choices that make this production-grade rather than tutorial-grade:

**`pyproject.toml` over `setup.cfg`** — Modern Python packaging standard (PEP 517/518). Future-proof and toolchain-compatible.

**Pydantic v2 for all data models** — `ContextEntry`, `ModelResponse`, `RunMetric`, `EvaluationReport` are all Pydantic models. Schema validation at the boundary, not buried in business logic.

**JSONL log format** — One JSON object per line. Streaming-friendly, grep-friendly, and crash-safe. A plain JSON array would corrupt the entire log on a mid-write crash.

**Pure function scorers** — Each of the 5 evaluation dimensions is a standalone function `(ContextEntry) → DimensionScore`. No shared state, no side effects. Easy to test, easy to add new dimensions.

**Strategy failures never crash the pipeline** — Every optimizer strategy is wrapped in try/except. A broken scorer returns a neutral score. The application continues. This is the difference between a library and a toy.

**Exit codes as a first-class feature** — `0/1/2` exit codes make the CLI a CI/CD citizen from day one. This was designed in, not bolted on.

**Lazy provider initialization** — SDK clients are created only when first needed and cached for reuse. A 3-provider `compare()` call reuses the same connections, not three separate SDK initializations.

---

## Roadmap

- [x] Phase 1 — Project scaffold, pyproject.toml, Makefile
- [x] Phase 2 — Context logger with token counting
- [x] Phase 3 — Multi-model router: Groq, Gemini, NVIDIA
- [x] Phase 4 — 5-dimension context evaluator
- [x] Phase 5 — Token monitor with cost tracking + percentiles
- [x] Phase 6 — Context optimizer: dedup, rerank, trim, compress
- [x] Phase 7 — Production CLI: run, compare, analyze, report, check
- [ ] Phase 8 — LangChain / LangGraph integration
- [ ] Phase 9 — Async support for high-throughput pipelines
- [ ] Phase 10 — Streamlit dashboard for visual monitoring

---

## Contributing

```bash
git clone https://github.com/Samruddhi-jadh/ContextLens.git
cd ContextLens
pip install -e ".[dev]"
make test      # run full test suite
make lint      # ruff + mypy
make format    # black
```

All contributions welcome. Open an issue first for major changes.

---

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Sam](https://github.com/Samruddhi-jadh) · Context engineering for production AI systems.

*If ContextLens helped you ship better AI, leave a star.*
