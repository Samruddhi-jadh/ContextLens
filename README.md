# ContextLens

**Context Engineering & Evaluation Toolkit for Deep AI Agents**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)

ContextLens is a lightweight, developer-focused toolkit that makes the
context passed to AI agents **visible, measurable, and automatically
improvable**.

---

## The Problem

Modern AI agents built with LangChain, LangGraph, or raw API calls
suffer from a common set of silent failures:

| Problem | Consequence |
|---|---|
| Redundant retrieved docs | Wasted tokens, diluted signal |
| Irrelevant context injected | Hallucinations, off-topic responses |
| Context window overflow | Silent API truncation or errors |
| No cost visibility | Surprise bills in production |
| No quality baseline | No way to know if your prompts are good |

ContextLens solves all five.

---

## Architecture
```
Developer prompt
      │
      ▼
┌─────────────────────────────────────────────────┐
│                  ContextLens                    │
│                                                 │
│  Logger → Evaluator → Optimizer → Monitor      │
│                                                 │
│         Multi-Model Router                      │
│    ┌──────────┬──────────┬──────────┐           │
│    │   Groq   │  Gemini  │  NVIDIA  │           │
│    └──────────┴──────────┴──────────┘           │
└─────────────────────────────────────────────────┘
      │
      ▼
  LensResult
  ├── response.text
  ├── context_entry  (logged, token-counted)
  ├── evaluation     (5-dimension score 0–100)
  └── metrics        (cost, latency, tokens)
```

---

## Installation
```bash
pip install contextlens
```

Or from source:
```bash
git clone https://github.com/yourusername/contextlens.git
cd contextlens
pip install -e ".[dev]"
cp .env.example .env   # add your API keys
```

### API Keys
```bash
# .env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
NVIDIA_API_KEY=your_nvidia_key     # optional
```

---

## Quickstart

### Python SDK
```python
from contextlens import ContextLens
from contextlens.core.logger import RetrievedDoc

lens = ContextLens(session_id="my_agent", budget_usd=1.00)

result = lens.run(
    prompt="Explain how context engineering reduces hallucinations.",
    system_prompt="You are a senior AI engineer. Be precise.",
    retrieved_docs=[
        RetrievedDoc(
            content="Context engineering involves...",
            source="internal_wiki",
            relevance_score=0.94,
        )
    ],
)

print(result.response.text)
print(f"Score:   {result.evaluation.overall_score}/100")
print(f"Cost:    ${result.response.estimated_cost_usd:.6f}")
print(f"Latency: {result.response.latency_ms:.0f}ms")
```

### Multi-provider comparison
```python
results = lens.compare(
    prompt="Explain attention in one paragraph.",
    providers=["groq", "gemini"],
)
# Prints a rich table sorted by latency
```

### Optimize bad context before re-running
```python
result = lens.run(prompt="...", retrieved_docs=many_docs)

if result.evaluation.overall_score < 70:
    opt = lens.optimize(result)
    print(opt.summary())          # tokens saved, strategies applied
    better = lens.run_entry(opt.optimized_entry)
```

---

## CLI
```bash
# Run a prompt with full observability
contextlens run "Explain transformers" --provider groq --optimize

# Compare providers side by side
contextlens compare "Write a Python async HTTP client" \
    --providers groq --providers gemini

# Evaluate a context log file (great for CI)
contextlens analyze logs/my_agent_20241201.jsonl --threshold 70

# Session cost + latency report
contextlens report --session my_agent --export report.json

# Health check all providers
contextlens check
```

---

## Evaluation Dimensions

Every context entry is scored 0–100 across 5 dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| **Length** | 20% | Token budget utilization (20–80% = ideal) |
| **Redundancy** | 25% | Cosine similarity across retrieved docs |
| **Relevance** | 25% | Term overlap between docs and prompt |
| **Specificity** | 15% | Prompt precision and constraint richness |
| **Completeness** | 15% | System prompt, role definition, history presence |

Grades: A (90+) · B (80+) · C (70+) · D (55+) · F (<55)

---

## Optimization Strategies

When evaluation finds problems, the optimizer applies targeted fixes:

| Strategy | Triggered by | What it does |
|---|---|---|
| **Deduplicate** | Redundancy < 80 | Removes near-duplicate docs via cosine similarity |
| **Re-rank** | Relevance < 75 | Sorts docs by relevance, keeps top-K |
| **Trim** | Length > 80% | Truncates long docs to token budget |
| **Compress** | Any | Sentence-level filtering by keyword relevance |

---

## CI/CD Integration

ContextLens exits with code `1` when context quality is below threshold.
Use this to gate deployments:
```yaml
# .github/workflows/context_quality.yml
- name: Check context quality
  run: |
    contextlens analyze logs/production_$(date +%Y%m%d).jsonl \
      --threshold 70
  # Exit 1 if any entry scores below 70
```

---

## Project Structure
```
contextlens/
├── contextlens/
│   ├── lens.py          # Main orchestrator
│   ├── config.py        # Pydantic settings
│   ├── core/
│   │   ├── logger.py    # Context capture (Phase 2)
│   │   ├── evaluator.py # Quality scoring (Phase 4)
│   │   ├── monitor.py   # Token + cost tracking (Phase 5)
│   │   └── optimizer.py # Automated repair (Phase 6)
│   ├── providers/
│   │   ├── base.py      # Abstract provider
│   │   ├── groq_provider.py
│   │   ├── gemini_provider.py
│   │   └── nvidia_provider.py
│   └── cli/
│       └── commands.py  # Typer CLI (Phase 7)
├── examples/
│   └── research_assistant.py
├── tests/
├── .env.example
├── pyproject.toml
└── Makefile
```

---

## Roadmap

- [x] Phase 1 — Project scaffold
- [x] Phase 2 — Context logger
- [x] Phase 3 — Multi-model router
- [x] Phase 4 — Context evaluator
- [x] Phase 5 — Token monitor
- [x] Phase 6 — Context optimizer
- [x] Phase 7 — CLI + production polish
- [ ] Phase 8 — Demo agent
- [ ] Phase 9 — GitHub deployment

- [ ] Phase 10 — Streamlit dashboard
=======


---

## Contributing
```bash
git clone https://github.com/yourusername/contextlens.git
cd contextlens
pip install -e ".[dev]"
make test   # run full test suite
make lint   # ruff + mypy
make format # black
```


---

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Sam](https://github.com/yourusername) · Context engineering for production AI systems.
```

---

### File 3 — `LICENSE`
```
MIT License

Copyright (c) 2025 Sam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=======

