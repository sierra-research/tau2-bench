# tau2-bench-agent Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-11-24

## Active Technologies
- Python 3.10+ (matches tau2-bench pyproject.toml requires-python) + pydantic (data models), loguru (logging), pathlib (file operations) (002-evaluation-store)
- Filesystem JSON files in `$TAU2_DATA_DIR` (default `./data`) (002-evaluation-store)
- Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`) (003-async-evaluation)
- Filesystem JSON (via 002-evaluation-store: `$TAU2_DATA_DIR/sessions/` for in-progress, `$TAU2_DATA_DIR/evaluations/` for completed) (003-async-evaluation)
- N/A (stateless utilities, state managed by 002-evaluation-store) (003-async-evaluation)
- Filesystem JSON (`$TAU2_DATA_DIR/evaluations/`) for post-hoc metrics emission (007-datadog-project)

- Python 3.10+ (per tau2-bench pyproject.toml requires-python) + httpx (>=0.28.0) for async HTTP client, a2a-sdk (>=0.3.12) with http-server extras for A2A protocol, loguru (>=0.7.3) for structured logging, pydantic for message validation (001-a2a-integration)

## Project Structure

```text
src/
tests/
```

## Commands

All Python commands must use `uv` as the package manager and runner:
- `uv run python <script>` - Run Python scripts
- `uv run pytest` - Run tests
- `uv add <package>` - Add dependencies
- `uv run ruff check .` - Run linting

## Code Style

Python 3.10+ (per tau2-bench pyproject.toml requires-python): Follow standard conventions

- Line length: 88 characters (configured in pyproject.toml)
- Import organization: Use Ruff's import sorting
- Type hints: Encouraged for new code, especially in core framework
- Docstrings: Required for public APIs and complex functions

## Recent Changes
- 007-datadog-project: Added Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`)
- 003-async-evaluation: Added Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`)
- 002-evaluation-store: Added Python 3.10+ (matches tau2-bench pyproject.toml requires-python) + pydantic (data models), loguru (logging), pathlib (file operations)


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
