.PHONY: help test lint format check clean

help:
	@echo "Available commands:"
	@echo "  make test      - Run tests with pytest"
	@echo "  make lint      - Run linting with ruff"
	@echo "  make format    - Auto-format code with ruff"
	@echo "  make check     - Run both linting and tests"
	@echo "  make clean     - Remove cache files and directories"

test:
	@echo "Running tests..."
	uv run pytest tests

lint:
	@echo "Running linter..."
	uv run ruff check src tests

format:
	@echo "Formatting code..."
	uv run ruff format src tests
	uv run ruff check --fix src tests

check: lint test
	@echo "✓ All checks passed!"

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✓ Cleanup complete!"

