# ============================================================================
# DEFAULT TARGETS - A2A/ADK Extension Code Only
# ============================================================================
# These targets work only on your A2A/ADK extension code to avoid
# modifying or testing the original tau2-bench repository.
# A2A/ADK paths: src/tau2/a2a/, tau2_agent/, tests/test_a2a_client/,
#                tests/test_adk_server/, tests/test_a2a_e2e/

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

## Run A2A/ADK tests only (default: excludes E2E, use test-e2e for E2E)
.PHONY: test
test:
	uv run pytest tests/test_a2a_client/ tests/test_adk_server/ -v

## Lint A2A/ADK code with ruff
.PHONY: lint
lint:
	uv run ruff check src/tau2/a2a/ tau2_agent/ tests/test_a2a_client/ tests/test_adk_server/ tests/test_a2a_e2e/

## Format A2A/ADK code with ruff
.PHONY: format
format:
	uv run ruff format src/tau2/a2a/ tau2_agent/ tests/test_a2a_client/ tests/test_adk_server/ tests/test_a2a_e2e/

## Lint and fix A2A/ADK code automatically
.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix src/tau2/a2a/ tau2_agent/ tests/test_a2a_client/ tests/test_adk_server/ tests/test_a2a_e2e/

## Check A2A/ADK code formatting without making changes
.PHONY: format-check
format-check:
	uv run ruff format --check src/tau2/a2a/ tau2_agent/ tests/test_a2a_client/ tests/test_adk_server/ tests/test_a2a_e2e/

## Run mypy type checker on A2A/ADK code (strict)
.PHONY: typecheck
typecheck:
	uv run mypy src/tau2/a2a/ tau2_agent/

## Run only E2E tests (requires ADK server running)
.PHONY: test-e2e
test-e2e:
	uv run pytest -m a2a_e2e -v

## Run A2A/ADK tests with coverage report
.PHONY: test-cov
test-cov:
	uv run pytest tests/test_a2a_client/ tests/test_adk_server/ tests/test_a2a_e2e/ --cov=src/tau2/a2a --cov=tau2_agent --cov-report=html --cov-report=term

## Run all quality checks (format-check, lint, typecheck, test) - A2A/ADK only
.PHONY: quality
quality: _quality-header _quality-format _quality-lint _quality-typecheck _quality-test _quality-footer

.PHONY: _quality-header
_quality-header:
	@echo "=== Running code quality checks (A2A/ADK module only) ==="

.PHONY: _quality-format
_quality-format:
	@echo "1. Checking code formatting..."
	@uv run ruff format --check src/tau2/a2a/ tau2_agent/ tests/test_a2a_client/ tests/test_adk_server/ tests/test_a2a_e2e/ || (echo "❌ Run 'make format' to fix." && exit 1)
	@echo "✅ Formatting OK"

.PHONY: _quality-lint
_quality-lint:
	@echo "2. Running linter..."
	@uv run ruff check src/tau2/a2a/ tau2_agent/ tests/test_a2a_client/ tests/test_adk_server/ tests/test_a2a_e2e/ || (echo "❌ Run 'make lint-fix' to fix." && exit 1)
	@echo "✅ Linter OK"

.PHONY: _quality-typecheck
_quality-typecheck:
	@echo "3. Running type checker..."
	@uv run mypy src/tau2/a2a/ tau2_agent/ || (echo "❌ Type check failed." && exit 1)
	@echo "✅ Types OK"

.PHONY: _quality-test
_quality-test:
	@echo "4. Running tests..."
	@uv run pytest tests/test_a2a_client/ tests/test_adk_server/ -q || (echo "❌ Tests failed." && exit 1)
	@echo "✅ Tests OK"

.PHONY: _quality-footer
_quality-footer:
	@echo "=== All quality checks passed! ==="

## Auto-fix linting and formatting issues on A2A/ADK code
.PHONY: fix
fix: lint-fix format
	@echo "✅ Auto-fixes applied to A2A/ADK code. Review changes before committing."

# ============================================================================
# REPO-WIDE TARGETS - Entire tau2-bench Repository
# ============================================================================
# These "-all" targets run commands on the entire repository, including
# the original tau2-bench code. Use with caution.

## Lint entire repository with ruff
.PHONY: lint-all
lint-all:
	uv run ruff check src/ tests/

## Format entire repository with ruff
.PHONY: format-all
format-all:
	uv run ruff format src/ tests/

## Check entire repository formatting without making changes
.PHONY: format-check-all
format-check-all:
	uv run ruff format --check src/ tests/

## Lint and fix entire repository automatically
.PHONY: lint-fix-all
lint-fix-all:
	uv run ruff check --fix src/ tests/

## Run mypy type checker on entire codebase (non-strict, may have errors)
.PHONY: typecheck-all
typecheck-all:
	uv run mypy src/ || true

## Run all tests in repository (includes original tau2-bench tests)
.PHONY: test-all
test-all:
	uv run pytest tests/ -v

## Run all tests with coverage report (entire repository)
.PHONY: test-cov-all
test-cov-all:
	uv run pytest tests/ --cov=src/tau2 --cov-report=html --cov-report=term

## Run all quality checks on entire repository (format-check, lint, typecheck, test)
.PHONY: quality-all
quality-all: _quality-all-header _quality-all-format _quality-all-lint _quality-all-typecheck _quality-all-test _quality-all-footer

.PHONY: _quality-all-header
_quality-all-header:
	@echo "=== Running code quality checks (ENTIRE REPOSITORY) ==="

.PHONY: _quality-all-format
_quality-all-format:
	@echo "1. Checking code formatting..."
	@uv run ruff format --check src/ tests/ || (echo "❌ Run 'make format-all' to fix." && exit 1)
	@echo "✅ Formatting OK"

.PHONY: _quality-all-lint
_quality-all-lint:
	@echo "2. Running linter..."
	@uv run ruff check src/ tests/ || (echo "❌ Run 'make lint-fix-all' to fix." && exit 1)
	@echo "✅ Linter OK"

.PHONY: _quality-all-typecheck
_quality-all-typecheck:
	@echo "3. Running type checker..."
	@uv run mypy src/ || (echo "⚠️ Type check errors (non-fatal)" && true)
	@echo "✅ Types checked"

.PHONY: _quality-all-test
_quality-all-test:
	@echo "4. Running all tests..."
	@uv run pytest tests/ -q || (echo "❌ Tests failed." && exit 1)
	@echo "✅ Tests OK"

.PHONY: _quality-all-footer
_quality-all-footer:
	@echo "=== All quality checks completed! ==="

## Auto-fix linting and formatting issues on entire repository
.PHONY: fix-all
fix-all: lint-fix-all format-all
	@echo "✅ Auto-fixes applied to entire repository. Review changes before committing."

# ============================================================================
# UTILITY TARGETS
# ============================================================================

## Start the Environment CLI for interacting with domain environments
.PHONY: env-cli
env-cli:
	uv run python -m tau2.environment.utils.interface_agent

## Install dependencies (using uv)
.PHONY: install
install:
	uv sync

## Display online help for commonly used targets in this Makefile
.PHONY: help
help:
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "  tau2-bench A2A/ADK Extension - Makefile Targets"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "DEFAULT TARGETS (A2A/ADK extension code only):"
	@echo "  make test          - Run A2A/ADK tests (excludes E2E)"
	@echo "  make test-e2e      - Run E2E tests only"
	@echo "  make test-cov      - Run A2A/ADK tests with coverage"
	@echo "  make lint          - Lint A2A/ADK code"
	@echo "  make format        - Format A2A/ADK code"
	@echo "  make lint-fix      - Lint and auto-fix A2A/ADK code"
	@echo "  make format-check  - Check A2A/ADK formatting (no changes)"
	@echo "  make typecheck     - Type check A2A/ADK code"
	@echo "  make quality       - Run all checks on A2A/ADK code"
	@echo "  make fix           - Auto-fix lint and format (A2A/ADK)"
	@echo ""
	@echo "REPO-WIDE TARGETS (entire tau2-bench repository):"
	@echo "  make test-all      - Run all tests (including original repo)"
	@echo "  make test-cov-all  - Run all tests with coverage"
	@echo "  make lint-all      - Lint entire repository"
	@echo "  make format-all    - Format entire repository"
	@echo "  make lint-fix-all  - Lint and auto-fix entire repository"
	@echo "  make format-check-all - Check entire repo formatting"
	@echo "  make typecheck-all - Type check entire repository"
	@echo "  make quality-all   - Run all checks on entire repository"
	@echo "  make fix-all       - Auto-fix lint and format (entire repo)"
	@echo ""
	@echo "UTILITY TARGETS:"
	@echo "  make install       - Install dependencies"
	@echo "  make clean         - Clean generated files"
	@echo "  make env-cli       - Start environment CLI"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
