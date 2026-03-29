# Makefile — dev shortcuts for ContextLens

.PHONY: install dev test lint format clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,dashboard]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check contextlens/
	mypy contextlens/

format:
	black contextlens/ tests/ examples/
	ruff check --fix contextlens/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info