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

## Run all tests
.PHONY: test
test:
	pytest tests/


## Start the Environment CLI for interacting with domain environments
.PHONY: env-cli
env-cli:
	python -m tau2.environment.utils.interface_agent

## Lint code with ruff
.PHONY: lint
lint:
	ruff check .

## Format code with ruff
.PHONY: format
format:
	ruff format .

## Lint and fix issues automatically
.PHONY: lint-fix
lint-fix:
	ruff check --fix .

## Run both linting and formatting
.PHONY: check-all
check-all: lint format

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
