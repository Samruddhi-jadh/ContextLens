# ContextLens
Context Engineering &amp; Evaluation Toolkit for Deep AI Agents

# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint (ruff)
        run: ruff check contextlens/

      - name: Type check (mypy)
        run: mypy contextlens/ --ignore-missing-imports

      - name: Run tests
        run: pytest tests/ -v --tb=short

  context-quality:
    name: Context Quality Gate
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install
        run: pip install -e .

      - name: Check context quality (if logs exist)
        run: |
          if ls logs/*.jsonl 1>/dev/null 2>&1; then
            contextlens analyze logs/*.jsonl --threshold 70
          else
            echo "No log files found — skipping quality gate."
          fi
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}