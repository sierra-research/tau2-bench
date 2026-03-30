# Default target
.PHONY: all
all: help

## Clean up generated files and virtual environment
.PHONY: clean
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build

## Run core tests (requires: uv sync --extra dev)
.PHONY: test
test:
	uv run pytest tests/ --ignore=tests/test_voice --ignore=tests/test_streaming --ignore=tests/test_gym --ignore=tests/test_domains/test_banking_knowledge

## Run voice and streaming tests (requires: uv sync --extra dev --extra voice)
.PHONY: test-voice
test-voice:
	uv run pytest tests/test_voice tests/test_streaming -m "not full_duplex_integration"

## Run knowledge/banking tests (requires: uv sync --extra dev --extra knowledge)
.PHONY: test-knowledge
test-knowledge:
	uv run pytest tests/test_domains/test_banking_knowledge

## Run gymnasium tests (requires: uv sync --extra dev --extra gym)
.PHONY: test-gym
test-gym:
	uv run pytest tests/test_gym

## Run all tests (requires: uv sync --all-extras)
.PHONY: test-all
test-all:
	uv run pytest tests/

## Start the Environment CLI for interacting with domain environments
.PHONY: env-cli
env-cli:
	uv run python -m tau2.environment.utils.interface_agent

## Lint code with ruff
.PHONY: lint
lint:
	uv run ruff check .

## Format code with ruff
.PHONY: format
format:
	uv run ruff format .

## Lint and fix issues automatically
.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix .

## Run both linting and formatting
.PHONY: check-all
check-all: lint format

## Install pre-commit hooks
.PHONY: setup-hooks
setup-hooks:
	uv run pre-commit install

## Display online help for commonly used targets in this Makefile
.PHONY: help
help:
	@awk '/^[a-zA-Z_\/\.0-9-]+:/ {        \
		nb = sub( /^## /, "", helpMsg );  \
		if (nb)                           \
			print  $$1 "\t" helpMsg;      \
	}                                     \
	{ helpMsg = $$0 }' $(MAKEFILE_LIST) | \
	column -ts $$'\t' |                   \
	expand -t 1 |                         \
	grep --color '^[^ ]*'
