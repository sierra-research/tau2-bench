# Pytest Configuration Reference

Configuration options, markers, and CI integration patterns.

---

## Configuration

### pyproject.toml

```toml
[tool.pytest.ini_options]
# Test discovery
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

# Async support
asyncio_mode = "strict"
asyncio_default_fixture_loop_scope = "function"

# Output
addopts = "-v --tb=short"

# Markers
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (mocked externals)",
    "e2e: End-to-end tests (real services)",
    "slow: Tests that take > 10 seconds",
]
```

### pytest.ini (alternative)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

asyncio_mode = strict
asyncio_default_fixture_loop_scope = function

markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (mocked externals)
    e2e: End-to-end tests (real services)
    slow: Tests that take > 10 seconds
```

---

## Markers

### Defining Markers

```python
# In test file
import pytest

@pytest.mark.unit
def test_fast_logic():
    """Runs with -m unit."""

@pytest.mark.integration
def test_component_interaction():
    """Runs with -m integration."""

@pytest.mark.e2e
class TestEndToEnd:
    """All methods run with -m e2e."""

@pytest.mark.slow
@pytest.mark.integration
def test_expensive_operation():
    """Has multiple markers."""
```

### Running by Marker

```bash
# Single marker
uv run pytest -m unit
uv run pytest -m e2e

# Exclude marker
uv run pytest -m "not slow"
uv run pytest -m "not e2e"

# Combine markers (AND)
uv run pytest -m "integration and not slow"

# Combine markers (OR)
uv run pytest -m "unit or integration"
```

### Built-in Markers

```python
# Skip unconditionally
@pytest.mark.skip(reason="Not implemented yet")

# Skip conditionally
@pytest.mark.skipif(sys.platform == "win32", reason="Unix only")

# Expected failure
@pytest.mark.xfail(reason="Known bug, fix pending")

# Parametrize
@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("", 0),
    ("a b c", 5),
])
def test_length(input, expected):
    assert len(input) == expected
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest

# Run specific file
uv run pytest tests/test_models.py

# Run specific class
uv run pytest tests/test_models.py::TestUser

# Run specific test
uv run pytest tests/test_models.py::TestUser::test_create

# Run by keyword
uv run pytest -k "create"
uv run pytest -k "create and not delete"
```

### Output Control

```bash
# Verbosity levels
uv run pytest -v      # Show test names
uv run pytest -vv     # Show more detail
uv run pytest -vvv    # Maximum verbosity

# Traceback styles
uv run pytest --tb=short   # Abbreviated
uv run pytest --tb=long    # Full with locals
uv run pytest --tb=no      # No traceback

# Show print output
uv run pytest -s
uv run pytest --capture=no

# Show slowest tests
uv run pytest --durations=10
```

### Failure Handling

```bash
# Stop on first failure
uv run pytest -x

# Stop after N failures
uv run pytest --maxfail=3

# Run last failed only
uv run pytest --lf

# Run failed first, then rest
uv run pytest --ff

# Debug on failure
uv run pytest --pdb
```

---

## Coverage

### Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
fail_under = 80
```

### Commands

```bash
# Run with coverage
uv run pytest --cov=src

# Generate HTML report
uv run pytest --cov=src --cov-report=html

# Show missing lines
uv run pytest --cov=src --cov-report=term-missing

# Fail if below threshold
uv run pytest --cov=src --cov-fail-under=80
```

---

## CI Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync

      - name: Run unit tests
        run: uv run pytest -m unit --timeout=30

      - name: Run integration tests
        run: uv run pytest -m integration --timeout=60

      - name: Run e2e tests
        run: uv run pytest -m e2e --timeout=300
        if: github.ref == 'refs/heads/main'
```

### Recommended CI Stages

```yaml
test:
  stages:
    - unit:        # Fast feedback (< 2 min)
        run: pytest -m unit --timeout=30
        on: [push, pull_request]

    - integration: # Medium feedback (< 5 min)
        run: pytest -m integration --timeout=60
        on: [push, pull_request]

    - e2e:         # Thorough validation (< 15 min)
        run: pytest -m e2e --timeout=300
        on: [push to main]
```

### Parallel Execution

```bash
# Install pytest-xdist
uv add pytest-xdist --dev

# Run tests in parallel
uv run pytest -n auto        # Auto-detect CPU count
uv run pytest -n 4           # Use 4 workers

# Distribute by file (faster for many files)
uv run pytest -n auto --dist=loadfile
```

---

## Useful Plugins

| Plugin | Purpose | Install |
|--------|---------|---------|
| pytest-asyncio | Async test support | `uv add pytest-asyncio --dev` |
| pytest-cov | Coverage reporting | `uv add pytest-cov --dev` |
| pytest-xdist | Parallel execution | `uv add pytest-xdist --dev` |
| pytest-timeout | Test timeouts | `uv add pytest-timeout --dev` |
| pytest-mock | Mocker fixture | `uv add pytest-mock --dev` |
| pytest-httpx | HTTPX mocking | `uv add pytest-httpx --dev` |

### Plugin Configuration

```toml
# pyproject.toml

# pytest-timeout
[tool.pytest.ini_options]
timeout = 60  # Default timeout per test

# pytest-asyncio
asyncio_mode = "strict"
```
