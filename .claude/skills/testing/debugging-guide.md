# Test Debugging Guide

Strategies for diagnosing and fixing test failures.

---

## Debugging Strategy

### Step 1: Isolate the Failure

**Never start with the full test suite.**

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
- Deadlocks (nested executors, shared resources)
- Blocking I/O without timeouts
- Event loop conflicts in async code

**Flaky Tests**:
- Race conditions
- Shared state between tests
- Timing-dependent assertions

**Slow Tests**:
- Profile with `pytest --durations=10`
- Unnecessary setup/teardown
- Repeated expensive operations

---

## Common Failure Patterns

### Pattern: Test Passes Alone, Fails in Suite

**Cause**: Shared state pollution

**Diagnosis**:
```bash
# Find which test causes pollution
uv run pytest tests/path/ --collect-only  # List test order
uv run pytest tests/path/test_a.py tests/path/test_b.py -v  # Run pairs
```

**Fix**: Use fixtures for isolation, avoid module-level mutables

### Pattern: Test Hangs Indefinitely

**Cause**: Blocking operation without timeout

**Diagnosis**:
```bash
# Force timeout
timeout 30 uv run pytest tests/path/test_file.py::test_hanging -v -s
```

**Fix**: Add timeouts to all I/O operations

```python
# Before
response = await client.get(url)

# After
response = await asyncio.wait_for(client.get(url), timeout=10.0)
```

### Pattern: Test Fails Intermittently

**Cause**: Race condition or timing issue

**Diagnosis**:
```bash
# Run multiple times
for i in {1..10}; do uv run pytest tests/path/test_file.py -v || break; done
```

**Fix**: Remove timing assumptions, use synchronization primitives

### Pattern: Test Fails on CI Only

**Cause**: Environment difference (resources, timing, paths)

**Diagnosis**:
- Check CI logs for resource constraints
- Compare environment variables
- Check for hardcoded paths

**Fix**: Make tests environment-agnostic

---

## Debugging Async Tests

### Event Loop Conflicts

```python
# BAD: Creates nested event loop
def test_sync_wrapper():
    asyncio.run(async_operation())  # May conflict with pytest-asyncio

# GOOD: Use async test
async def test_async_operation():
    await async_operation()
```

### Unclosed Resources

```python
# BAD: Resource leak
async def test_client():
    client = httpx.AsyncClient()
    response = await client.get(url)
    assert response.status_code == 200
    # Client never closed!

# GOOD: Context manager
async def test_client():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        assert response.status_code == 200
```

### Task Cancellation

```python
# Ensure tasks are awaited or cancelled
async def test_with_background_task():
    task = asyncio.create_task(background_work())
    try:
        result = await main_operation()
        assert result.success
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
```

---

## Debugging Session Template

Use this template when investigating test failures:

```markdown
## Test Failure Investigation

**Test**: `tests/path/test_file.py::TestClass::test_method`
**Symptom**: [timeout | assertion error | exception]

### Step 1: Reproduce
```bash
uv run pytest tests/path/test_file.py::TestClass::test_method -v -s
```

### Step 2: Isolate
- Fails in isolation? [yes/no]
- Fails with other tests? [yes/no]
- Timing-dependent? [yes/no]

### Step 3: Evidence
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

## Useful Debugging Commands

```bash
# Show slowest tests
uv run pytest --durations=10

# Stop on first failure
uv run pytest -x

# Stop after N failures
uv run pytest --maxfail=3

# Run last failed tests only
uv run pytest --lf

# Run failed tests first, then rest
uv run pytest --ff

# Increase verbosity
uv run pytest -vvv

# Show local variables in tracebacks
uv run pytest --tb=long

# Drop into debugger on failure
uv run pytest --pdb

# Show test names as they run
uv run pytest -v --tb=no
```
