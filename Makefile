.PHONY: setup setup-ml lint format test check

setup:            ## Install core + dev dependencies
	uv sync

setup-ml:         ## Also install the deep-learning stack (torch, timm, transformers)
	uv sync --extra ml

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

check: lint test  ## What CI and agents run before committing
