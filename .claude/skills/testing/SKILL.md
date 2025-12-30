---
name: testing
description: |
  Testing skill for pytest tests, test debugging, and test infrastructure.

  TRIGGER KEYWORDS: test, tests, testing, pytest, unit test, integration test,
  e2e test, end-to-end test, test failure, failing test, test error, broken test,
  flaky test, slow test, hanging test, test coverage, coverage gap, mock, mocking,
  fixture, fixtures, conftest, assert, assertion, TDD, test driven, write tests,
  add tests, create tests, run tests, debug test, fix test, test case, test suite.

  Use this skill when:
  - Writing, creating, or adding new tests
  - Debugging failing, hanging, or broken tests
  - Investigating flaky or intermittent test failures
  - Analyzing or improving test coverage
  - Setting up test infrastructure (fixtures, conftest.py, markers)
  - Reviewing test quality or test anti-patterns
  - Running or configuring pytest
  - Creating mocks, stubs, or test doubles
  - Fixing slow or timing-dependent tests

  Enforces behavior-focused testing with pytest, FIRST principles, and AAA pattern.
---

# Testing Skill

## Purpose

Guide systematic test creation and debugging. Tests validate behavior, not implementation details.

---

## Core Principles (FIRST)

Based on Uncle Bob Martin's FIRST principles for effective unit testing.

### 1. Fast

Tests must run quickly. Slow tests don't get run.

- Unit tests: < 1 second each
- Full unit suite: < 2 minutes
- Slow tests block feedback loops and discourage TDD

### 2. Independent (Isolated)

Tests must not affect each other. Run in any order, get same results.

- No shared mutable state between tests
- Each test sets up its own preconditions
- Each test cleans up after itself

### 3. Repeatable

Same test, same result. Every time, on every machine.

- No timing-dependent assertions
- No reliance on external service availability
- No environment-specific assumptions (paths, ports, configs)
- Control randomness with seeds

### 4. Self-Validating

Tests have a clear pass/fail result. No manual inspection required.

```python
# BAD: Requires manual inspection
def test_output():
    result = process(data)
    print(result)  # "Check if this looks right"

# GOOD: Automatic pass/fail
def test_output():
    result = process(data)
    assert result == expected_output
```

### 5. Thorough

Test the full behavior space, not just the happy path.

- Success cases AND failure cases
- Boundary conditions (empty, null, max, min)
- Edge cases and error handling
- State transitions

---

## Behavioral Testing Principles

### Test Behavior, Not Implementation

Tests verify **what** code does, not **how** it does it. Implementation can change; behavior contracts should not.

```python
# BAD: Tests internal implementation
def test_cache_uses_dict():
    cache = Cache()
    assert isinstance(cache._storage, dict)

# GOOD: Tests observable behavior
def test_cache_returns_stored_value():
    cache = Cache()
    cache.set("key", "value")
    assert cache.get("key") == "value"
```

### One Behavior Per Test

Each test should verify one logical behavior. Failures should pinpoint the problem.

```python
# BAD: One test does everything
def test_full_workflow():
    # 100 lines of setup, action, and assertions

# GOOD: Focused tests with clear purpose
def test_create_succeeds_with_valid_data(): ...
def test_create_rejects_empty_name(): ...
def test_update_preserves_created_timestamp(): ...
```

### Test the Unhappy Path

Every success path needs corresponding failure tests.

```python
# Both are required
def test_login_succeeds_with_valid_credentials():
    result = login("user", "correct_password")
    assert result.success

def test_login_fails_with_wrong_password():
    with pytest.raises(AuthenticationError):
        login("user", "wrong_password")
```

---

## Test Hierarchy

### Unit Tests (`tests/test_<module>/`)

**Purpose**: Test logic in isolation

| Characteristic | Requirement |
|----------------|-------------|
| Execution time | < 1 second per test |
| Dependencies | All mocked |
| Coverage | High (80%+), including edge cases |
| Naming | After the module they test |

```
tests/test_<module>/
    test_<module>.py           # Main module tests
    test_<submodule>.py        # Submodule tests
```

```python
# tests/test_store/test_models.py
class TestEvaluationSession:
    def test_create_with_valid_data(self):
        """Session creation accepts valid inputs."""

    def test_create_rejects_invalid_status(self):
        """Validation rejects unknown status values."""
```

### Integration Tests (`tests/test_<feature>/`)

**Purpose**: Test component interactions

| Characteristic | Requirement |
|----------------|-------------|
| External services | Mocked |
| Data flow | Verified across boundaries |
| Error handling | Tested at integration points |
| Naming | After the feature |

```
tests/test_<feature>/
    conftest.py               # Shared fixtures
    test_<flow>.py            # Flow-specific tests
```

```python
# tests/test_a2a_integration/test_message_flow.py
class TestA2AMessageFlow:
    async def test_message_sent_and_response_parsed(self, mock_http_client):
        """Full message cycle works with mocked HTTP."""
```

### E2E Tests (`tests/test_<feature>_e2e/`)

**Purpose**: Validate real behavior

| Characteristic | Requirement |
|----------------|-------------|
| External services | Real when possible |
| Execution time | Longer (acceptable) |
| Scope | Critical paths only |
| Naming | `_e2e` suffix |

```
tests/test_<feature>_e2e/
    conftest.py               # Server fixtures, credentials
    test_<scenario>.py        # Scenario-based tests
```

```python
# tests/test_datadog_e2e/test_observability_flow.py
@pytest.mark.e2e
class TestObservabilityFlow:
    async def test_traces_appear_in_datadog(self, live_agent):
        """Traces from agent calls appear in Datadog within 30s."""
```

**Running E2E tests**: Use parallel execution for faster results:
```bash
uv run pytest -m "a2a_e2e and not smoke" -n 4 --dist=loadfile  # ~1:41
uv run pytest -m "smoke" -n 2 --dist=loadfile
uv run pytest -m "datadog_e2e"  # Serial only (session-scoped)
```

---

## Anti-Patterns

### 1. Testing Implementation Details

```python
# BAD: Asserts on private attributes
assert obj._internal_counter == 5
assert mock_db.execute.call_count == 3

# GOOD: Asserts on observable outcomes
assert obj.get_count() == 5
assert len(results) == 3
```

### 2. Overly Broad Tests

```python
# BAD: Tests everything, failure reveals nothing
def test_user_workflow():
    user = create_user()
    user.update_profile(...)
    user.add_payment(...)
    user.place_order(...)
    assert user.orders[0].status == "complete"

# GOOD: Each test has one reason to fail
def test_create_user_assigns_id(): ...
def test_update_profile_changes_name(): ...
def test_place_order_requires_payment(): ...
```

### 3. Missing Edge Cases (Happy Path Only)

```python
# BAD: Only tests happy path
def test_parse_json():
    assert parse('{"key": "value"}') == {"key": "value"}

# GOOD: Tests boundaries
def test_parse_empty_object(): ...
def test_parse_nested_structure(): ...
def test_parse_invalid_json_raises(): ...
def test_parse_empty_string_raises(): ...
```

### 4. Magic Numbers

```python
# BAD: Unexplained values
assert response.status_code == 200
assert len(result) == 42

# GOOD: Named or explained
assert response.status_code == HTTPStatus.OK
assert len(result) == len(input_items)  # One result per input
```

### 5. Shared Mutable State

```python
# BAD: Tests affect each other
_test_cache = {}  # Module-level mutable

def test_first():
    _test_cache["key"] = "value"

def test_second():
    assert "key" not in _test_cache  # Fails if test_first runs first

# GOOD: Fixture provides isolated state
@pytest.fixture
def cache():
    return {}

def test_first(cache):
    cache["key"] = "value"
```

### 6. Time-Dependent Assertions

```python
# BAD: Flaky on slow machines
start = time.time()
do_operation()
assert time.time() - start < 0.1

# GOOD: Mock time or use generous bounds
def test_operation_completes(mocker):
    mock_time = mocker.patch("time.time")
    # Control time explicitly
```

### 7. Over-Mocking (Mockery)

When you mock so much that you're testing the mocks, not the code.

```python
# BAD: Everything is mocked, testing nothing real
def test_process_data(mocker):
    mock_validator = mocker.patch("app.validator")
    mock_transformer = mocker.patch("app.transformer")
    mock_saver = mocker.patch("app.saver")
    mock_validator.return_value = True
    mock_transformer.return_value = {"data": "transformed"}

    result = process_data(input)

    # Only testing that mocks were called, not actual behavior
    mock_validator.assert_called_once()
    mock_transformer.assert_called_once()

# GOOD: Mock external dependencies, test real logic
def test_process_data(mocker):
    mocker.patch("app.external_api.send")  # Only mock external

    result = process_data(valid_input)

    assert result.status == "processed"
    assert result.transformed_field == expected_value
```

### 8. The Liar (Missing/Weak Assertions)

Tests that pass but don't actually verify anything meaningful.

```python
# BAD: No real assertion
def test_process():
    result = process(data)
    assert result is not None  # Passes even if completely wrong

# BAD: Tests that something was called, not that it worked
def test_send_email(mocker):
    mock_send = mocker.patch("app.send_email")
    notify_user(user)
    mock_send.assert_called()  # Called, but with what? Did it work?

# GOOD: Verify actual outcomes
def test_process():
    result = process(data)
    assert result.status == "success"
    assert result.output == expected_output

# GOOD: Verify call arguments matter
def test_send_email(mocker):
    mock_send = mocker.patch("app.send_email")
    notify_user(user)
    mock_send.assert_called_once_with(
        to=user.email,
        subject="Notification",
        body=mocker.ANY,
    )
```

### 9. Free Ride (Piggyback)

Adding unrelated assertions to existing tests instead of writing new ones.

```python
# BAD: Test does too many unrelated things
def test_create_user():
    user = create_user("alice", "alice@example.com")
    assert user.name == "alice"
    assert user.email == "alice@example.com"
    # Unrelated assertions piggybacking
    assert validate_email("alice@example.com") == True
    assert hash_password("password123") != "password123"

# GOOD: Separate tests for separate behaviors
def test_create_user_sets_name(): ...
def test_create_user_sets_email(): ...
def test_validate_email_accepts_valid(): ...
def test_hash_password_is_irreversible(): ...
```

### 10. Local Hero (Environment-Dependent)

Tests that only pass on the original developer's machine.

```python
# BAD: Hardcoded paths and environment
def test_load_config():
    config = load_config("/Users/alice/project/config.json")
    assert config["debug"] == True

# BAD: Assumes specific port is available
def test_server():
    server = start_server(port=8080)  # What if 8080 is in use?

# GOOD: Use fixtures and relative paths
def test_load_config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"debug": true}')
    config = load_config(config_file)
    assert config["debug"] == True

# GOOD: Use dynamic port assignment
def test_server():
    server = start_server(port=0)  # OS assigns available port
    assert server.is_running
```

### 11. Slow Poke

Tests that take too long and block the feedback loop.

```python
# BAD: Sleeps and real waits
def test_retry_logic():
    time.sleep(5)  # Waiting for real timeout
    result = operation_with_retry()
    assert result.success

# BAD: Unnecessary I/O in unit tests
def test_data_processing():
    data = load_from_remote_api()  # Slow network call
    assert process(data) == expected

# GOOD: Mock time and external calls
def test_retry_logic(mocker):
    mocker.patch("time.sleep")  # Don't actually sleep
    result = operation_with_retry()
    assert result.success

# GOOD: Use fixtures with test data
def test_data_processing(sample_data):
    assert process(sample_data) == expected
```

### 12. Bad Test Names (Enumerator)

Names that don't describe what's being tested.

```python
# BAD: Meaningless names
def test1(): ...
def test2(): ...
def test_it(): ...
def test_stuff(): ...
def test_user(): ...  # What about the user?

# GOOD: Describe behavior and condition
def test_user_creation_assigns_unique_id(): ...
def test_user_creation_fails_with_duplicate_email(): ...
def test_user_deletion_removes_associated_data(): ...
```

---

## Fixture Best Practices

### Scope Appropriately

```python
@pytest.fixture(scope="session")
def database_connection():
    """Expensive resource shared across all tests."""

@pytest.fixture(scope="function")  # Default
def clean_state():
    """Fresh state for each test."""

@pytest.fixture(scope="class")
def shared_context():
    """Shared across test class methods."""
```

### Always Clean Up

```python
@pytest.fixture
def server():
    proc = subprocess.Popen(cmd)
    yield proc
    proc.terminate()
    proc.wait(timeout=10)
```

### Handle Async Properly

```python
@pytest.fixture
async def async_client():
    async with httpx.AsyncClient() as client:
        yield client
```

---

## Coverage Guidelines

| Test Type | Target | Focus |
|-----------|--------|-------|
| Unit | 80%+ | All logic branches |
| Integration | Key paths | Critical flows |
| E2E | Happy + error | Real behavior |

### Blind Spots to Cover

- **Error paths**: Exception handling, not just success
- **Edge cases**: Empty, null, max values, unicode
- **Concurrency**: Race conditions, deadlocks
- **State transitions**: All valid state changes

---

## Quick Reference

### Test Naming

```python
def test_<unit>_<behavior>_<condition>():
    """<Unit> <behavior> when <condition>."""

# Examples
def test_parser_returns_empty_dict_for_empty_input(): ...
def test_validator_rejects_negative_values(): ...
def test_cache_expires_entries_after_ttl(): ...
```

### Test Structure (Arrange-Act-Assert)

```python
def test_example():
    # Arrange: Set up preconditions
    user = User(name="test")

    # Act: Perform the action
    result = user.greet()

    # Assert: Verify the outcome
    assert result == "Hello, test"
```

### What Every Feature Needs

- [ ] Happy path (normal operation)
- [ ] Error conditions (invalid input, missing data)
- [ ] Edge cases (empty, null, max values)
- [ ] State transitions (all valid paths)
- [ ] Concurrency (if applicable)
- [ ] Resource cleanup (no leaks)

---

## Additional Resources

- For debugging strategies, see [debugging-guide.md](debugging-guide.md)
- For pytest configuration, see [pytest-reference.md](pytest-reference.md)
