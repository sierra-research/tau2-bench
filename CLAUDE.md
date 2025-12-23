# tau2-bench-agent Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-11-24

## Active Technologies
- Python 3.10+ (matches tau2-bench pyproject.toml requires-python) + pydantic (data models), loguru (logging), pathlib (file operations) (002-evaluation-store)
- Filesystem JSON files in `$TAU2_DATA_DIR` (default `./data`) (002-evaluation-store)
- Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`) (003-async-evaluation)
- Filesystem JSON (via 002-evaluation-store: `$TAU2_DATA_DIR/sessions/` for in-progress, `$TAU2_DATA_DIR/evaluations/` for completed) (003-async-evaluation)
- N/A (stateless utilities, state managed by 002-evaluation-store) (003-async-evaluation)

- Python 3.10+ (per tau2-bench pyproject.toml requires-python) + httpx (>=0.28.0) for async HTTP client, a2a-sdk (>=0.3.12) with http-server extras for A2A protocol, loguru (>=0.7.3) for structured logging, pydantic for message validation (001-a2a-integration)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.10+ (per tau2-bench pyproject.toml requires-python): Follow standard conventions

## Recent Changes
- 003-async-evaluation: Added Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`)
- 003-async-evaluation: Added Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`)
- 002-evaluation-store: Added Python 3.10+ (matches tau2-bench pyproject.toml requires-python) + pydantic (data models), loguru (logging), pathlib (file operations)


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
