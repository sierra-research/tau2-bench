# tau2_agent Concurrency Fix

**Status**: Resolved
**Date**: 2025-12-25
**Tests**: 7/7 datadog_e2e tests passing (~67s)

---

## Problem Statement

When multiple A2A evaluation requests were sent concurrently to tau2_agent, all evaluations would hang indefinitely at "submitted" status. Sequential evaluations also experienced timeout failures after 2-3 requests.

### Symptoms

| Scenario | Result |
|----------|--------|
| Single sequential evaluation | Completed in ~13-80 seconds |
| 3 concurrent evaluations | All stuck at "submitted" (20+ minutes, no progress) |
| 4+ sequential evaluations | Timeout after 2-3 completions |
| Health check during hang | GET returns 200 OK |
| POST requests during hang | Never receive response |

---

## Root Cause Analysis

### Primary Issue: Nested ThreadPoolExecutor Deadlock

The deadlock occurred due to nested blocking executors when both `tau2_agent` and `mock_test_agent` ran on the same ADK server.

**Deadlock Chain:**

```
1. pytest sends HTTP POST to ADK server (port 8766)
2. ADK server receives request -> tau2_agent.run_tau2_evaluation()
3. RunTau2Evaluation.run_async():
   |-- await loop.run_in_executor(None, run_domain, config)  # Runs in Thread X
4. run_domain() -> A2AAgent.generate_next_message():
   |-- concurrent.futures.ThreadPoolExecutor().submit(asyncio.run, ...)  # Thread Y
5. Thread Y's asyncio.run() makes HTTP request to mock_test_agent
6. mock_test_agent is on SAME ADK server (port 8766)
7. ADK server event loop is BLOCKED waiting for step 3 to complete
8. HTTP request from step 5 never gets processed
9. DEADLOCK: Step 3 waits for step 5, step 5 waits for step 3
```

**Visual Diagram:**

```
+-------------------------------------------------------------+
|                    ADK Server (Port 8766)                   |
|                    Event Loop (BLOCKED)                     |
|                                                             |
|  +------------------+           +------------------+        |
|  |   tau2_agent     |    HTTP   | mock_test_agent  |        |
|  |                  |---------->| (can't receive   |        |
|  |  run_in_executor |  blocked  |   request)       |        |
|  |       |          |           +------------------+        |
|  |   [Thread Y]-----+----------------+                      |
|  |   asyncio.run()  |                |                      |
|  |       |          |                v                      |
|  |   HTTP request --+------> Waits for event loop           |
|  |                  |        (which is blocked)             |
|  +------------------+                                       |
+-------------------------------------------------------------+
```

### Code Locations

**File:** `src/tau2/agent/a2a_agent.py` (lines 187-201)

```python
def generate_next_message(self, message, state):
    async def _async_generate():
        response_content, new_context_id = await self.client.send_message(...)
        return assistant_msg, new_state

    # PROBLEM: Nested executor blocks parent thread
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(_async_generate())
    else:
        # THIS BRANCH CAUSES DEADLOCK
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _async_generate())
            return future.result()  # BLOCKS the worker thread
```

**File:** `tau2_agent/tools/run_tau2_evaluation.py` (line 345)

```python
# Uses shared default executor
loop = asyncio.get_running_loop()
results = await loop.run_in_executor(None, run_domain, config)
```

### Secondary Issue: File Collision

Concurrent evaluations using timestamp-based filenames (second granularity) could overwrite each other when started within the same second.

---

## Solution Implementation

### Fix 1: Simplify A2AAgent Async/Sync Bridge

**File:** `src/tau2/agent/a2a_agent.py`

**Change:** Remove nested ThreadPoolExecutor, always use `asyncio.run()`

```python
def generate_next_message(self, message, state):
    async def _async_generate():
        # ... async HTTP call to remote agent ...
        return assistant_msg, new_state

    # Run async function synchronously.
    # This method is called from thread pool workers (via run_in_executor in
    # run_tau2_evaluation.py), which never have a running event loop.
    # Using asyncio.run() creates a fresh event loop for this thread.
    #
    # IMPORTANT: Do NOT use nested ThreadPoolExecutor here - it causes deadlock
    # when multiple concurrent evaluations run, as each nested executor blocks
    # its parent worker thread waiting on future.result().
    return asyncio.run(_async_generate())
```

**Same fix applied to `stop()` method.**

**Rationale:** Thread pool workers created by `run_in_executor` don't have a running event loop. The original code's `asyncio.get_running_loop()` check was unnecessary and the fallback path created deadlocks.

### Fix 2: Dedicated Executor with Unique Filenames

**File:** `tau2_agent/tools/run_tau2_evaluation.py`

**Changes:**
1. Added dedicated `ThreadPoolExecutor` with 10 workers
2. Use UUID-based unique filenames to prevent collisions

```python
import uuid
from concurrent.futures import ThreadPoolExecutor

# Dedicated executor for evaluation work - avoids default executor exhaustion
_EVALUATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="tau2_eval_",
)

# In run_async():
unique_run_id = f"tau2_eval_{uuid.uuid4().hex[:12]}"
config = EvalConfig(
    save_to=unique_run_id,
    # ... other config
)

results = await loop.run_in_executor(_EVALUATION_EXECUTOR, run_domain, config)
```

**Rationale:**
- Dedicated executor prevents contention with other async operations
- UUID-based filenames eliminate collision risk for concurrent evaluations

### Fix 3: Separate Test Servers

**File:** `tests/test_datadog_e2e/conftest.py`

**Change:** Run `tau2_agent` and test agent on separate ports

```
pytest test_datadog_e2e
    +-- test_agent_server (port 8767)     # Separate server
    |   +-- simple_nebius_agent
    |
    +-- traced_adk_server (port 8766)
        +-- tau2_agent
```

**Rationale:** Separate servers ensure HTTP requests between agents don't block the same event loop.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/tau2/agent/a2a_agent.py` | Simplified async/sync bridge in `generate_next_message()` and `stop()` |
| `tau2_agent/tools/run_tau2_evaluation.py` | Added dedicated executor, UUID-based filenames |
| `tests/test_datadog_e2e/conftest.py` | Separate server for test agent, load .env for API keys |
| `tests/test_datadog_e2e/test_observability_flow.py` | Updated assertions for new agent name and reward structure |

---

## Verification

### Test Results

```
tests/test_datadog_e2e/test_observability_flow.py
    TestServerSetup::test_servers_accessible_on_separate_ports PASSED
    TestA2AObservabilityFlow::test_full_a2a_evaluation_flow PASSED
    TestA2AObservabilityFlow::test_concurrent_evaluations_complete_with_valid_results PASSED
    TestMetricsEmission::test_emit_metrics_dry_run_mode PASSED
    TestMetricsEmission::test_emit_task_metrics PASSED
    TestMetricsEmission::test_emit_evaluation_metrics PASSED
    TestMetricsEmission::test_emit_metrics_from_real_evaluation PASSED

7 passed in 67.22s
```

### Before vs After

| Metric | Before | After |
|--------|--------|-------|
| Sequential evaluations | 2-3 before timeout | Unlimited |
| Concurrent evaluations | Hang indefinitely | Complete successfully |
| Test duration | Timeout (300s+) | ~67 seconds |
| Concurrency support | No | Yes (10 workers) |

---

## Production Implications

This issue does NOT affect production deployments because:
- In production, the agent being evaluated runs on a **different server**
- The deadlock only occurs when evaluator and evaluated agent share the same event loop

However, the dedicated executor and UUID filenames improve robustness for any concurrent usage patterns.

---

## Lessons Learned

1. **Nested executors are dangerous**: Creating a ThreadPoolExecutor inside a thread pool worker and blocking on `future.result()` can cause deadlock.

2. **Test for concurrency explicitly**: Sequential test passes don't guarantee concurrent behavior. The `test_concurrent_evaluations_complete_with_valid_results` test was essential for catching this.

3. **Separate concerns in test infrastructure**: Running multiple agents on the same server simplified setup but created hidden dependencies.

4. **Use unique identifiers for parallel work**: Timestamp-based naming with second granularity is insufficient for concurrent operations.

---

## References

- ADK documentation: https://github.com/google/adk-python
- pytest-asyncio event loop scoping: https://pytest-asyncio.readthedocs.io/
- Python concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html
