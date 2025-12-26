# A2A Test Suite Refactor Plan

> **Reference Implementation:** The `tests/test_datadog_e2e/` suite demonstrates best practices for E2E testing. While this suite is specific to Datadog observability and won't be part of the A2A integration feature, it should be used as a guide for test design patterns.

## Problem Statement

The current A2A test suite has several issues:
1. **`test_a2a_endpoint.py`** - Tests pass with 404 responses, providing false confidence
2. **Missing E2E coverage** - No tests for multiple concurrent evaluations with store persistence
3. **SSE streaming not validated** - E2E tests don't verify SSE event format

## Root Cause Analysis

### `test_a2a_endpoint.py` Issues

**Issue 1: Wrong endpoint path**
```python
# OLD (broken) - posts to root, gets 404
response = await client.post("/", json=a2a_message)

# CORRECT - A2A endpoints are at /a2a/{agent_name}/
A2A_ENDPOINT = "/a2a/tau2_agent"
response = await client.post(A2A_ENDPOINT, json=a2a_message)
```

**Issue 2: Relative path doesn't work**
```python
# OLD (broken) - relative path doesn't resolve correctly
app = get_fast_api_app(agents_dir="./tau2_agent", web=False, a2a=True)

# CORRECT - use absolute path
PROJECT_ROOT = Path(__file__).parent.parent.parent
app = get_fast_api_app(agents_dir=str(PROJECT_ROOT), web=False, a2a=True)
```

**Issue 3: Tests accept 404 as valid**
```python
# OLD (broken) - silently passes on 404
assert response.status_code in [200, 400, 404, 501]

# CORRECT - require 200 for valid requests
assert response.status_code == 200
```

**Issue 4: Redirect handling**
```python
# POST to /a2a/tau2_agent/ (trailing slash) returns 307 redirect
# Need to either use no trailing slash or enable follow_redirects
async_client = AsyncClient(
    transport=ASGITransport(app=adk_app),
    base_url="http://test",
    follow_redirects=True,
)
```

---

## Current Test Inventory

### Already Working Well
| Directory | Purpose | Status |
|-----------|---------|--------|
| `tests/test_adk_server/test_tools.py` | Unit tests for RunTau2Evaluation with mocked tau2 | ✅ Good |
| `tests/test_a2a_client/` | A2A protocol client tests with MockA2ATransport | ✅ Good |
| `tests/test_a2a_e2e/test_evaluation_flow.py` | A2A JSON-RPC protocol compliance | ✅ Good |
| `tests/test_a2a_e2e/test_smoke.py` | Single evaluation smoke tests | ✅ Partial |

### Needs Fixing
| File | Issue |
|------|-------|
| `tests/test_adk_server/test_a2a_endpoint.py` | Wrong paths, accepts 404 |

### Needs Adding
| Test | Purpose |
|------|---------|
| Multiple concurrent evaluations | Verify async eval handling |
| EvaluationStore persistence in E2E | Verify store integration end-to-end |
| SSE streaming format validation | Verify event stream to client |

---

## Phase 1: Fix `test_a2a_endpoint.py`

**File:** `tests/test_adk_server/test_a2a_endpoint.py`

### Changes Made (in progress)

1. Use absolute path via `Path(__file__).parent.parent.parent`
2. Use correct A2A endpoint path `/a2a/tau2_agent`
3. Add `async_client` fixture with `follow_redirects=True`
4. Remove 404-accepting assertions
5. Add proper test for agent card accessibility
6. Add test for invalid method error handling

### Updated Test File Structure

```python
from pathlib import Path
from google.adk.cli.fast_api import get_fast_api_app
from httpx import ASGITransport, AsyncClient

PROJECT_ROOT = Path(__file__).parent.parent.parent
A2A_AGENT_CARD = "/a2a/tau2_agent/.well-known/agent-card.json"
A2A_ENDPOINT = "/a2a/tau2_agent"  # No trailing slash for POST

@pytest.fixture
def adk_app():
    return get_fast_api_app(agents_dir=str(PROJECT_ROOT), web=False, a2a=True)

@pytest.fixture
def async_client(adk_app):
    return AsyncClient(
        transport=ASGITransport(app=adk_app),
        base_url="http://test",
        follow_redirects=True,
    )
```

---

## Phase 2: Add E2E Tests to `tests/test_a2a_e2e/`

### 2.1 Add concurrent evaluations test
**File:** `tests/test_a2a_e2e/test_smoke.py`

```python
@pytest.mark.asyncio
async def test_multiple_concurrent_evaluations(self, adk_server, smoke_tool_context):
    """Test multiple async evaluations can run concurrently."""
    import asyncio
    from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

    tool = RunTau2Evaluation(name="run_tau2_evaluation", description="...")

    async def run_eval(eval_id: int):
        events = []
        async for event in tool.run_async(
            args={
                "domain": "mock",
                "agent_endpoint": adk_server,
                "num_tasks": 1,
                "num_trials": 1,
            },
            tool_context=smoke_tool_context,
        ):
            events.append(event)
        return events

    # Run 3 concurrent evaluations
    results = await asyncio.gather(
        run_eval(1), run_eval(2), run_eval(3),
        return_exceptions=True
    )

    # All should complete successfully
    for i, r in enumerate(results):
        assert not isinstance(r, Exception), f"Eval {i} failed: {r}"
        assert len(r) >= 2, f"Eval {i} missing events"
```

### 2.2 Add EvaluationStore persistence test
**File:** `tests/test_a2a_e2e/test_smoke.py`

```python
@pytest.mark.asyncio
async def test_evaluation_persists_to_store(self, adk_server, smoke_tool_context, tmp_path):
    """Verify evaluation results persist to EvaluationStore."""
    import os
    from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

    # Use temp directory for store
    os.environ["TAU2_DATA_DIR"] = str(tmp_path)
    (tmp_path / "sessions").mkdir()
    (tmp_path / "evaluations").mkdir()

    tool = RunTau2Evaluation(name="run_tau2_evaluation", description="...")

    async for event in tool.run_async(
        args={"domain": "mock", "agent_endpoint": adk_server, "num_tasks": 1},
        tool_context=smoke_tool_context,
    ):
        pass  # Consume all events

    # Verify evaluation file created
    eval_files = list((tmp_path / "evaluations").glob("*.json"))
    assert len(eval_files) == 1, "Evaluation should persist to store"

    # Verify structure
    import json
    with open(eval_files[0]) as f:
        data = json.load(f)
    assert data["status"] == "completed"
    assert "results" in data
```

### 2.3 Add SSE streaming format test
**File:** `tests/test_a2a_e2e/test_evaluation_flow.py`

```python
@pytest.mark.asyncio
async def test_e2e_sse_streaming_format(adk_server):
    """Test that SSE streaming returns properly formatted events."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "sse-test-001",
                    "role": "user",
                    "parts": [{"text": "Run a mock evaluation"}],
                }
            },
            "id": "req-sse-001",
        }

        async with client.stream(
            "POST",
            f"{adk_server}/",
            json=jsonrpc_request,
            headers={"Accept": "text/event-stream"},
        ) as response:
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(line[5:].strip())

            assert len(events) > 0, "Should receive SSE events"
```

---

## Phase 3: Add conftest fixture for Store Testing

**File:** `tests/test_a2a_e2e/conftest.py`

```python
@pytest.fixture
def isolated_store_dir(tmp_path):
    """Create isolated data directory for EvaluationStore."""
    import os

    sessions_dir = tmp_path / "sessions"
    evals_dir = tmp_path / "evaluations"
    sessions_dir.mkdir()
    evals_dir.mkdir()

    old_dir = os.environ.get("TAU2_DATA_DIR")
    os.environ["TAU2_DATA_DIR"] = str(tmp_path)

    yield tmp_path

    # Restore
    if old_dir:
        os.environ["TAU2_DATA_DIR"] = old_dir
    else:
        del os.environ["TAU2_DATA_DIR"]
```

---

## Files to Modify

| File | Action |
|------|--------|
| `tests/test_adk_server/test_a2a_endpoint.py` | Fix paths, remove 404 acceptance |
| `tests/test_a2a_e2e/conftest.py` | Add `isolated_store_dir` fixture |
| `tests/test_a2a_e2e/test_smoke.py` | Add concurrent evaluations + store persistence tests |
| `tests/test_a2a_e2e/test_evaluation_flow.py` | Add SSE streaming format test |

---

## Test Commands

```bash
# Run unit + mock tests (default, runs in CI)
uv run pytest

# Run A2A E2E tests (requires running agent)
uv run pytest -m "a2a_e2e"

# Run smoke tests (requires API keys)
uv run pytest -m "smoke"
```

---

## Reference: `test_datadog_e2e` Suite Analysis

The `tests/test_datadog_e2e/` suite is the gold standard for E2E testing in this project. Below is a detailed analysis of its patterns that should guide the A2A E2E test implementation.

### Architecture Overview

```
tests/test_datadog_e2e/
├── conftest.py                  # Fixtures for traced server + helpers
└── test_observability_flow.py   # E2E test classes
```

### Key Design Patterns

#### 1. TracedServer Dataclass
**File:** `conftest.py:32-49`

Encapsulates all server state in a clean dataclass:

```python
@dataclass
class TracedServer:
    process: subprocess.Popen    # Server process handle
    data_dir: Path               # Isolated data directory
    endpoint: str                # Base URL (e.g., "http://localhost:8766")
    tau2_agent_endpoint: str     # Full A2A path

    @property
    def evaluations_dir(self) -> Path:
        return self.data_dir / "evaluations"

    @property
    def mock_agent_endpoint(self) -> str:
        return f"{self.endpoint}/a2a/mock_test_agent"
```

**Pattern:** Use dataclass to bundle related server info with computed properties.

#### 2. Session-Scoped Isolated Data Directory
**File:** `conftest.py:64-85`

```python
@pytest.fixture(scope="session")
def temp_data_dir(tmp_path_factory):
    """Create isolated temporary data directory for the test session."""
    data_dir = tmp_path_factory.mktemp("tau2_data")
    (data_dir / "sessions").mkdir(exist_ok=True)
    (data_dir / "evaluations").mkdir(exist_ok=True)

    # Symlink to project's domain task files
    source_tau2_dir = PROJECT_ROOT / "data" / "tau2"
    if source_tau2_dir.exists():
        (data_dir / "tau2").symlink_to(source_tau2_dir)

    return data_dir
```

**Pattern:**
- Use `tmp_path_factory` for session-scoped temp directories
- Pre-create required subdirectories
- Symlink read-only project data to avoid duplication

#### 3. Subprocess Server Management
**File:** `conftest.py:88-224`

```python
@pytest.fixture(scope="session")
def traced_adk_server(temp_data_dir):
    # 1. Check port availability
    if is_port_in_use(ADK_SERVER_PORT):
        pytest.fail(f"Port {ADK_SERVER_PORT} is in use...")

    # 2. Configure environment
    env = os.environ.copy()
    env["DD_TRACE_ENABLED"] = "true"
    env["TAU2_DATA_DIR"] = str(temp_data_dir)

    # 3. Start subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        preexec_fn=os.setsid if os.name != "nt" else None,  # Process group
    )

    # 4. Health check loop
    while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
        try:
            response = httpx.get(agent_card_url, timeout=2)
            if response.status_code == 200:
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.5)

    # 5. Yield server info
    yield TracedServer(process=process, data_dir=temp_data_dir, ...)

    # 6. Cleanup with process group kill
    finally:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
```

**Pattern:**
- Port availability check before starting
- Environment-based configuration
- Process group for clean shutdown (`preexec_fn=os.setsid`)
- Health check via agent card endpoint
- Graceful shutdown with SIGTERM, fallback to SIGKILL

#### 4. SSE Streaming Helper
**File:** `conftest.py:280-354`

```python
async def send_a2a_evaluation_request(
    endpoint: str,
    domain: str = "mock",
    agent_endpoint: str = "http://mock-agent:8000",
    num_tasks: int = 2,
    num_trials: int = 1,
    stream: bool = True,
    timeout: float = 120.0,
) -> AsyncIterator[dict]:
    """Send A2A evaluation request and stream SSE events."""
    request = build_a2a_evaluation_request(...)

    if stream:
        request["method"] = "message/stream"
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", endpoint, json=request,
                                      headers={"Accept": "text/event-stream"}) as response:
                buffer = ""
                async for chunk in response.aiter_text():
                    # Parse SSE events from buffer
                    yield parse_sse_event(event_text)
```

**Pattern:**
- Async generator for SSE event streaming
- Configurable streaming vs single POST
- Buffer-based SSE parsing
- Proper timeout configuration

#### 5. SSE Event Parser
**File:** `conftest.py:357-389`

```python
def parse_sse_event(event_text: str) -> dict | None:
    """Parse SSE event: 'event: type\ndata: {...}'"""
    lines = event_text.strip().split("\n")
    event_type = None
    data_lines = []

    for line in lines:
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    data = json.loads("".join(data_lines))
    if event_type:
        data["_event_type"] = event_type
    return data
```

**Pattern:** Standard SSE parsing with event type metadata.

### Test Class Organization

#### TestA2AObservabilityFlow
**File:** `test_observability_flow.py:27-228`

```python
@pytest.mark.datadog_e2e
class TestA2AObservabilityFlow:
    """Tests for A2A evaluation with Datadog tracing integration."""

    async def test_traced_server_starts_with_ddtrace(self, traced_adk_server):
        """Health check - verify server accessible."""

    async def test_evaluation_tool_persists_to_store(self, traced_adk_server):
        """Core flow - A2A request → SSE events → store persistence."""

    async def test_emit_metrics_processes_stored_evaluation(self, traced_adk_server):
        """Pipeline - A2A → Store → emit_metrics.py processing."""

    async def test_event_state_progression(self, traced_adk_server):
        """SSE events - verify state progression."""

    async def test_trace_context_metadata_in_events(self, traced_adk_server):
        """Metadata - verify trace context in stored evaluation."""
```

**Pattern:**
- Class groups related tests
- Single marker at class level (`@pytest.mark.datadog_e2e`)
- Progressive complexity: health → core flow → full pipeline
- Each test verifies one aspect of the flow

### Store Persistence Verification Pattern

```python
async def test_evaluation_tool_persists_to_store(self, traced_adk_server):
    os.environ["TAU2_DATA_DIR"] = str(traced_adk_server.data_dir)

    events_collected = []
    async for event in send_a2a_evaluation_request(
        endpoint=traced_adk_server.tau2_agent_endpoint,
        domain="mock",
        agent_endpoint=traced_adk_server.mock_agent_endpoint,
        num_tasks=1,
        num_trials=1,
        stream=True,
    ):
        events_collected.append(event)

    # Verify events received
    assert len(events_collected) > 0

    # Verify files created
    eval_files = list(traced_adk_server.evaluations_dir.glob("*.json"))
    assert len(eval_files) > 0

    # Verify JSON structure
    with open(eval_files[-1]) as f:
        eval_data = json.load(f)

    assert eval_data["status"] == "completed"
    assert "results" in eval_data
    assert eval_data["domain"] == "mock"
```

**Pattern:**
1. Set environment for store
2. Stream SSE events and collect
3. Verify events received
4. Verify files created in store directory
5. Verify JSON structure of stored data

### Failed Evaluation Handling

```python
async def test_failed_evaluation_persisted(self, traced_adk_server):
    # Count before
    sessions_before = list((data_dir / "sessions").glob("*.json"))

    # Send request that will fail
    async for _ in send_a2a_evaluation_request(
        domain="invalid_domain_xyz",  # Will fail
        ...
    ):
        pass

    # Check for new files
    sessions_after = list((data_dir / "sessions").glob("*.json"))
    new_sessions = [s for s in sessions_after if s not in sessions_before]

    if new_sessions:
        with open(new_sessions[-1]) as f:
            session_data = json.load(f)
        assert session_data.get("status") in ("failed", "in_progress")
```

**Pattern:** Test failure paths by triggering known errors and verifying they're recorded.

---

## Applying datadog_e2e Patterns to a2a_e2e

### Recommended Changes to `tests/test_a2a_e2e/conftest.py`

The existing `test_a2a_e2e/conftest.py` already follows many patterns. Add:

```python
# 1. Add isolated store fixture (from datadog_e2e pattern)
@pytest.fixture
def isolated_store_dir(tmp_path):
    """Create isolated data directory for EvaluationStore."""
    sessions_dir = tmp_path / "sessions"
    evals_dir = tmp_path / "evaluations"
    sessions_dir.mkdir()
    evals_dir.mkdir()

    old_dir = os.environ.get("TAU2_DATA_DIR")
    os.environ["TAU2_DATA_DIR"] = str(tmp_path)

    yield tmp_path

    if old_dir:
        os.environ["TAU2_DATA_DIR"] = old_dir
    elif "TAU2_DATA_DIR" in os.environ:
        del os.environ["TAU2_DATA_DIR"]

# 2. Add SSE streaming helper (adapted from datadog_e2e)
async def send_a2a_message(
    endpoint: str,
    message: str,
    stream: bool = False,
    timeout: float = 30.0,
) -> AsyncIterator[dict]:
    """Send A2A message and optionally stream SSE events."""
    # ... (similar to datadog_e2e pattern)
```

### Recommended Test Structure

```python
@pytest.mark.a2a_e2e
class TestA2AEvaluationFlow:
    """E2E tests for A2A evaluation flow."""

    async def test_server_health(self, adk_server):
        """Verify server accessible via agent card."""

    async def test_single_evaluation_completes(self, adk_server, isolated_store_dir):
        """Single evaluation request completes and persists."""

    async def test_multiple_concurrent_evaluations(self, adk_server, isolated_store_dir):
        """Multiple async evaluations run concurrently."""

    async def test_sse_streaming_events(self, adk_server):
        """SSE events stream properly during evaluation."""

    async def test_evaluation_store_persistence(self, adk_server, isolated_store_dir):
        """Results persist to EvaluationStore."""

    async def test_failed_evaluation_recorded(self, adk_server, isolated_store_dir):
        """Failed evaluations are recorded in store."""
```

---

## Success Criteria

1. `test_a2a_endpoint.py` returns 200 for valid requests (not 404)
2. E2E tests verify multiple concurrent evaluations complete
3. E2E tests verify EvaluationStore persistence end-to-end
4. E2E tests verify SSE streaming format
5. Clear separation: `a2a_mock` runs by default, `a2a_e2e` is opt-in
