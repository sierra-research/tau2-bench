---
name: testing
description: |
  Testing skill for creating, debugging, and maintaining tests. Use when:
  - Creating new tests for implemented features
  - Debugging failing or hanging tests
  - Analyzing test coverage gaps
  - Investigating flaky or slow tests
  - Setting up test infrastructure (fixtures, conftest)
  Provides strategies for unit, integration, and e2e testing with pytest.
---

# Testing Skill

## Purpose

Guide systematic test creation and debugging to catch implementation issues early. Tests should validate:
1. **Unit tests**: Individual function/class logic with high coverage
2. **Integration tests**: Component interactions with mocks where appropriate
3. **E2E tests**: Real behavior against actual services when necessary

---

## Test Hierarchy

### Unit Tests (`tests/test_<module>/`)

**Purpose**: Test implemented logic in isolation

**Characteristics**:
- Fast execution (< 1 second per test)
- No external dependencies (mocked)
- High coverage of edge cases
- Named after the module they test

**Naming Convention**:
```
tests/test_<module>/
    test_<module>.py           # Main module tests
    test_<submodule>.py        # Submodule tests
```

**Example**:
```python
# tests/test_store/test_models.py - tests src/tau2/store/models.py
class TestEvaluationSession:
    def test_create_with_valid_data(self):
        """Test session creation with valid inputs."""

    def test_create_rejects_invalid_status(self):
        """Test validation rejects unknown status values."""

    def test_update_status_transitions_correctly(self):
        """Test status can only transition in valid order."""
```

### Integration Tests (`tests/test_<feature>/`)

**Purpose**: Test component interactions

**Characteristics**:
- Mock external services (APIs, databases)
- Test data flow between components
- Verify error handling across boundaries
- Named after the feature being tested

**Naming Convention**:
```
tests/test_<feature>/
    conftest.py               # Shared fixtures
    test_<flow>.py            # Flow-specific tests
```

**Example**:
```python
# tests/test_a2a_integration/test_message_flow.py
class TestA2AMessageFlow:
    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client for A2A requests."""

    async def test_message_sent_and_response_parsed(self, mock_http_client):
        """Test full message send/receive cycle with mocked HTTP."""
```

### E2E Tests (`tests/test_<feature>_e2e/`)

**Purpose**: Test real behavior against actual services

**Characteristics**:
- Use real external services when possible
- Longer execution time (acceptable)
- Catch integration issues missed by mocks
- Named with `_e2e` suffix

**Naming Convention**:
```
tests/test_<feature>_e2e/
    conftest.py               # Server fixtures, API keys
    test_<scenario>.py        # Scenario-based tests
```

**Example**:
```python
# tests/test_datadog_e2e/test_observability_flow.py
@pytest.mark.datadog_e2e
class TestA2AObservabilityFlow:
    async def test_concurrent_evaluations_complete(self, traced_server):
        """Test multiple evaluations run concurrently without deadlock."""
```

---

## Test Debugging Strategy

### Step 1: Isolate the Failure

**Never start with the full test suite for debugging.**

```bash
# Run single test first
uv run pytest tests/path/test_file.py::TestClass::test_method -v

# Run single class
uv run pytest tests/path/test_file.py::TestClass -v

# Run single file
uv run pytest tests/path/test_file.py -v
```

### Step 2: Add Visibility

**For hanging tests or unclear failures:**

```bash
# Show print statements
uv run pytest ... -s

# Show all output including captured
uv run pytest ... -v --capture=no

# Set shorter timeout to fail fast
timeout 30 uv run pytest ... -v
```

**For subprocess-based tests (servers):**

```python
# In conftest.py - capture server output
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # Combine streams
    text=True,
)

# On failure, print server output
@pytest.fixture
def server(request):
    proc = start_server()
    yield proc
    if request.node.rep_call.failed:
        stdout, _ = proc.communicate(timeout=5)
        print(f"Server output:\n{stdout}")
```

### Step 3: Progressive Scope Expansion

After single test passes:

```bash
# Expand to related tests
uv run pytest tests/path/test_file.py -v

# Expand to test directory
uv run pytest tests/path/ -v

# Only then run full suite
uv run pytest -v
```

### Step 4: Identify Root Cause Patterns

**Hanging Tests**:
- Check for deadlocks (nested executors, shared resources)
- Check for blocking I/O without timeouts
- Check for event loop conflicts in async code

**Flaky Tests**:
- Check for race conditions
- Check for shared state between tests
- Check for timing-dependent assertions

**Slow Tests**:
- Profile with `pytest --durations=10`
- Check for unnecessary setup/teardown
- Check for repeated expensive operations

---

## Fixture Best Practices

### Scope Appropriately

```python
# Session scope for expensive shared resources
@pytest.fixture(scope="session")
def database_connection():
    """Single DB connection for all tests."""

# Function scope for isolated state
@pytest.fixture(scope="function")  # Default
def clean_state():
    """Fresh state for each test."""

# Class scope for related tests
@pytest.fixture(scope="class")
def shared_context():
    """Shared across test class methods."""
```

### Clean Up Resources

```python
@pytest.fixture
def server():
    proc = subprocess.Popen(cmd)
    yield proc
    # Always clean up
    proc.terminate()
    proc.wait(timeout=10)
```

### Handle Async Properly

```python
@pytest.fixture
async def async_client():
    async with httpx.AsyncClient() as client:
        yield client
    # Client closed automatically
```

---

## Coverage Guidelines

### Minimum Coverage Targets

| Test Type | Target | Rationale |
|-----------|--------|-----------|
| Unit tests | 80%+ | Core logic fully tested |
| Integration | Key paths | Critical flows covered |
| E2E | Happy path + errors | Real behavior validated |

### Coverage Blind Spots to Avoid

1. **Error paths**: Test exception handling, not just success
2. **Edge cases**: Empty inputs, max values, unicode
3. **Concurrency**: Parallel execution, race conditions
4. **State transitions**: All valid state changes

### Check Coverage

```bash
# Run with coverage
uv run pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

---

## Common Anti-Patterns

### 1. Testing Implementation, Not Behavior

```python
# BAD: Tests internal implementation
def test_cache_uses_dict():
    cache = Cache()
    assert isinstance(cache._storage, dict)

# GOOD: Tests behavior
def test_cache_returns_stored_value():
    cache = Cache()
    cache.set("key", "value")
    assert cache.get("key") == "value"
```

### 2. Overly Long Test Methods

```python
# BAD: One test does everything
def test_full_workflow():
    # 100 lines of setup, action, and assertions

# GOOD: Focused tests
def test_create_succeeds():
    ...

def test_update_after_create():
    ...

def test_delete_after_update():
    ...
```

### 3. Missing Negative Tests

```python
# Need both positive AND negative tests
def test_create_with_valid_data():
    result = create(valid_data)
    assert result.success

def test_create_with_invalid_data():
    with pytest.raises(ValidationError):
        create(invalid_data)
```

### 4. Hardcoded Test Values

```python
# BAD: Magic numbers
assert response.status_code == 200
assert len(result) == 42

# GOOD: Named constants or explicit
assert response.status_code == HTTPStatus.OK
assert len(result) == expected_count
```

---

## Test Discovery Checklist

When implementing a feature, ensure tests exist for:

- [ ] Happy path (normal operation)
- [ ] Error conditions (invalid input, missing data)
- [ ] Edge cases (empty, null, max values)
- [ ] State transitions (all valid paths)
- [ ] Concurrency (if applicable)
- [ ] Resource cleanup (no leaks)
- [ ] Integration points (external calls)

---

## Debugging Session Template

When debugging test failures, follow this template:

```markdown
## Test Failure Investigation

**Test**: `tests/path/test_file.py::TestClass::test_method`
**Symptom**: [Describe what happens - timeout, assertion error, exception]

### Step 1: Reproduce
```bash
uv run pytest tests/path/test_file.py::TestClass::test_method -v -s
```

### Step 2: Isolate
- Does it fail in isolation? [yes/no]
- Does it fail with other tests? [yes/no]
- Is it timing-dependent? [yes/no]

### Step 3: Gather Evidence
- Error output: [paste]
- Relevant logs: [paste]
- State at failure: [describe]

### Step 4: Root Cause
[Describe the actual cause]

### Step 5: Fix
[Describe the fix and why it works]

### Step 6: Verify
```bash
uv run pytest tests/path/ -v  # Verify in broader context
```
```

---

## Pytest Configuration Reference

### pytest.ini / pyproject.toml

```ini
[pytest]
# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Async support
asyncio_mode = strict
asyncio_default_fixture_loop_scope = function

# Markers for selective running
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (mocked externals)
    e2e: End-to-end tests (real services)
    slow: Tests that take > 10 seconds
```

### Running Subsets

```bash
# By marker
uv run pytest -m unit
uv run pytest -m "not slow"
uv run pytest -m "integration or e2e"

# By keyword
uv run pytest -k "concurrent"
uv run pytest -k "not flaky"
```

---

## Integration with CI

### Recommended CI Test Stages

```yaml
test:
  stages:
    - unit:        # Fast feedback (< 2 min)
        run: pytest -m unit --timeout=30
    - integration: # Medium feedback (< 5 min)
        run: pytest -m integration --timeout=60
    - e2e:         # Thorough validation (< 15 min)
        run: pytest -m e2e --timeout=300
```

### Fail Fast in Development

```bash
# Stop on first failure
uv run pytest -x

# Stop after N failures
uv run pytest --maxfail=3
```
