# Tasks: Shared SSE Streaming Utilities

**Input**: Design documents from `/specs/003-async-evaluation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Unit tests included per spec.md (FR-007 success criteria require testing ADK conversion)

**Organization**: Tasks grouped by functional requirement from spec.md to enable independent implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1=Streaming Module, US2=RunTau2Evaluation Integration)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and streaming module structure

- [X] T001 Create streaming module directory structure at `tau2_agent/streaming/`
- [X] T002 Create `tau2_agent/streaming/__init__.py` with public exports placeholder
- [X] T003 [P] Create test directory structure at `tests/test_streaming/`
- [X] T004 [P] Create `tests/test_streaming/__init__.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core type definitions that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Define `TaskState` type alias (`Literal["submitted", "working", "completed", "failed"]`) in `tau2_agent/streaming/events.py`
- [X] T006 [P] Create `tau2_agent/streaming/metadata.py` with tau2 namespace constants (TAU2_STATE, TAU2_PROGRESS, TAU2_EVALUATION_ID, etc.)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Streaming Utilities Module (Priority: P0)

**Goal**: Provide EvaluationProgress dataclass and event builder functions for SSE streaming

**Independent Test**: Unit tests verify progress calculation and event creation produce valid ADK Event objects

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T007 [P] [US1] Create test file `tests/test_streaming/test_progress.py` with tests for EvaluationProgress:
  - `test_percent_zero_tasks` - 0 tasks returns 0%
  - `test_percent_calculation` - correct percentage at various stages
  - `test_elapsed_seconds` - elapsed time calculation
  - `test_increment` - completed_tasks increments and task_id updates
  - `test_to_metadata` - produces correct tau2.* namespaced dict

- [X] T008 [P] [US1] Create test file `tests/test_streaming/test_events.py` with tests for event builders:
  - `test_create_adk_progress_event_submitted` - produces Event with state=submitted
  - `test_create_adk_progress_event_working` - includes progress metadata
  - `test_create_adk_progress_event_with_progress_object` - integrates EvaluationProgress
  - `test_create_adk_error_event` - produces Event with error_code set
  - `test_create_adk_result_event` - includes results in content
  - `test_required_metadata_present` - tau2.state, tau2.progress, tau2.evaluation_id present

### Implementation for User Story 1

- [X] T009 [US1] Implement `EvaluationProgress` dataclass in `tau2_agent/streaming/progress.py`:
  - Fields: `total_tasks`, `completed_tasks`, `current_task_id`, `current_trial`, `total_trials`, `started_at`
  - Properties: `percent`, `elapsed_seconds`
  - Methods: `to_metadata()`, `increment(task_id)`

- [X] T010 [US1] Implement `create_adk_progress_event()` in `tau2_agent/streaming/events.py`:
  - Parameters: `invocation_id`, `state`, `message`, `evaluation_id`, `progress`, `**extra_metadata`
  - Returns: ADK `Event` with tau2 metadata in `custom_metadata`
  - Required metadata: `tau2.state`, `tau2.progress`, `tau2.evaluation_id`

- [X] T011 [US1] Implement `create_adk_error_event()` in `tau2_agent/streaming/events.py`:
  - Parameters: `invocation_id`, `evaluation_id`, `error_message`, `error_code`, `**extra_metadata`
  - Returns: ADK `Event` with `error_code` field set and state="failed"

- [X] T012 [US1] Implement `create_adk_result_event()` in `tau2_agent/streaming/events.py`:
  - Parameters: `invocation_id`, `evaluation_id`, `results`, `message`, `**extra_metadata`
  - Returns: ADK `Event` with results dict in content and state="completed"

- [X] T013 [US1] Update `tau2_agent/streaming/__init__.py` to export public API:
  - `EvaluationProgress` from progress.py
  - `create_adk_progress_event`, `create_adk_error_event`, `create_adk_result_event` from events.py
  - `TaskState` type alias

- [X] T014 [US1] Run tests for User Story 1 and verify all pass: `pytest tests/test_streaming/ -v`

**Checkpoint**: Streaming utilities module is fully functional and tested independently

---

## Phase 4: User Story 2 - RunTau2Evaluation Integration (Priority: P0)

**Goal**: Update existing RunTau2Evaluation tool to use streaming utilities for SSE progress and expose trace context

**Independent Test**: Tool emits SSE progress events during evaluation; trace context (evaluation_id, task_id) is accessible

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T015 [P] [US2] Add streaming tests to `tests/test_adk_server/test_tools.py`:
  - `test_run_tau2_evaluation_yields_events` - tool yields Event objects (not just dict)
  - `test_run_tau2_evaluation_submitted_event` - first event has state="submitted"
  - `test_run_tau2_evaluation_progress_events` - working events emitted during evaluation
  - `test_run_tau2_evaluation_result_event` - final event contains results
  - `test_run_tau2_evaluation_error_event` - failed state on error

- [X] T016 [P] [US2] Add trace context test to `tests/test_adk_server/test_tools.py`:
  - `test_run_tau2_evaluation_trace_context` - events contain evaluation_id, task_id, domain for OTel instrumentation

### Implementation for User Story 2

- [X] T017 [US2] Modify `tau2_agent/tools/run_tau2_evaluation.py` to import streaming utilities:
  - Import `EvaluationProgress`, `create_adk_progress_event`, `create_adk_error_event`, `create_adk_result_event`
  - Import `Event` from `google.adk.events.event`

- [X] T018 [US2] Change RunTau2Evaluation.run_async() return type to `AsyncIterator[Event]`:
  - Update method signature to be async generator (use `yield` instead of `return`)
  - Update type hints

- [X] T019 [US2] Add streaming events to RunTau2Evaluation._execute():
  - Initialize `EvaluationProgress(total_tasks=num_tasks)`
  - Yield `create_adk_progress_event(state="submitted")` at start
  - Yield `create_adk_progress_event(state="working")` per task completion (requires callback integration)
  - Yield `create_adk_result_event()` with aggregated results at end
  - Wrap errors with `create_adk_error_event()`

- [X] T020 [US2] Expose trace context in events for 007-datadog instrumentation:
  - Include `tau2.evaluation_id`, `tau2.domain`, `tau2.current_task_id`, `tau2.agent_endpoint` in all events
  - Document trace context keys in code comments

- [X] T021 [US2] Run tests for User Story 2 and verify all pass: `pytest tests/test_adk_server/test_tools.py -v -k "run_tau2_evaluation"`

**Checkpoint**: RunTau2Evaluation emits SSE progress events and exposes trace context for instrumentation

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Code quality, validation, and documentation per CONTRIBUTING.md standards

### Code Quality (per CONTRIBUTING.md)

- [X] T022 [P] Run Ruff linting and fix issues: `make lint-fix` on `tau2_agent/streaming/` and `tests/test_streaming/`
- [X] T023 [P] Run Ruff formatting: `make format` on `tau2_agent/streaming/` and `tests/test_streaming/`
- [X] T024 [P] Verify all code quality checks pass: `make check-all`
- [X] T025 [P] Verify type hints present on all public functions in `tau2_agent/streaming/events.py` and `tau2_agent/streaming/progress.py`
- [X] T026 [P] Verify docstrings present on all public APIs: `EvaluationProgress`, `create_adk_progress_event`, `create_adk_error_event`, `create_adk_result_event`

### Validation

- [X] T027 [P] Run full test suite to ensure no regressions: `pytest tests/ -v`
- [X] T028 [P] Validate streaming events against contracts schema: verify events match `contracts/streaming-events.yaml` format
- [X] T029 Run quickstart.md validation - verify examples from quickstart.md work correctly

---

## Phase 6: Integration Tests - SSE Streaming End-to-End (Priority: P1)

**Goal**: Verify streaming events flow correctly through ADK server and A2A protocol, not just in isolation

**Context**: Phase 4 unit tests use mocks to verify event structure. Integration tests are needed to ensure:
1. Events are actually emitted through the ADK server's SSE transport
2. A2A clients can receive and process streaming events in real-time
3. The AsyncIterator from `run_async()` integrates correctly with ADK's event loop

### Background: How ADK Streaming Works

The ADK server handles `AsyncIterator[Event]` returns from tools by:
1. Tool's `run_async()` yields `Event` objects
2. ADK's `ToolContext` collects yielded events
3. Events are serialized and sent via SSE to connected clients
4. A2A protocol wraps SSE in `tasks/sendSubscribe` response stream

Key files for reference:
- `tau2_agent/tools/run_tau2_evaluation.py` - Tool implementation (yields events)
- `tau2_agent/adk_server.py` - ADK server setup
- `.venv/.../google/adk/` - ADK internals for event handling
- `.venv/.../a2a/server/` - A2A server SSE implementation

### Tests for Phase 6

> **NOTE**: These tests require actual server instances, not mocks

- [X] T030 [P] [INT] Create integration test directory `tests/test_streaming_integration/`

- [X] T031 [P] [INT] Create `tests/test_streaming_integration/conftest.py` with fixtures:
  - `adk_test_server` - Spawns actual ADK server on random port (use `pytest-asyncio`)
  - `mock_tau2_backend` - Mocks `tau2.run.run_domain` to return quickly without LLM calls
  - `sse_client` - HTTP client configured for SSE (consider `httpx-sse` or `aiohttp`)

- [X] T032 [INT] Create `tests/test_streaming_integration/test_sse_streaming.py`:
  - `test_sse_events_received_in_order` - Client receives submitted→working→completed in sequence
  - `test_sse_event_timing` - Events are streamed incrementally (not batched at end)
  - `test_sse_connection_stays_open_during_evaluation` - Connection persists until completed/failed
  - `test_sse_reconnection_receives_remaining_events` - (stretch goal) SSE reconnection works

- [X] T033 [INT] Create `tests/test_streaming_integration/test_a2a_streaming.py`:
  - `test_a2a_task_subscribe_streams_events` - A2A `tasks/sendSubscribe` returns event stream
  - `test_a2a_event_metadata_preserved` - tau2.* metadata survives A2A serialization
  - `test_a2a_error_event_closes_stream` - Failed state terminates stream gracefully

- [X] T034 [INT] Create `tests/test_streaming_integration/test_thread_context.py`:
  - `test_events_have_consistent_invocation_id` - All events in one call share invocation_id
  - `test_evaluation_id_unique_per_invocation` - Different calls get different evaluation_ids
  - `test_tool_context_receives_all_events` - Verify ToolContext accumulates yielded events

### Implementation Notes

**Server Setup Pattern**:
```python
@pytest.fixture
async def adk_test_server():
    """Spawn ADK server for integration testing."""
    import asyncio
    from tau2_agent.adk_server import create_app

    app = create_app()
    # Use test server (e.g., httpx.ASGITransport or actual uvicorn)
    async with AsyncTestServer(app) as server:
        yield server.base_url
```

**SSE Client Pattern**:
```python
async def collect_sse_events(url: str, timeout: float = 30.0) -> list[dict]:
    """Collect all SSE events from endpoint."""
    events = []
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:]))
    return events
```

**Mock Backend Pattern**:
```python
@pytest.fixture
def mock_tau2_backend(monkeypatch):
    """Fast mock that yields controllable results."""
    async def fake_run_domain(*args, **kwargs):
        # Return quickly with predictable results
        return create_mock_results(num_tasks=3)

    monkeypatch.setattr("tau2.run.run_domain", fake_run_domain)
```

### Acceptance Criteria

1. **T032**: SSE client receives at least 3 events (submitted, working, completed) in correct order
2. **T033**: A2A protocol test verifies event metadata survives JSON serialization
3. **T034**: Thread context test confirms invocation_id consistency across all events

### Dependencies

- Requires: Phase 4 complete (streaming implementation exists)
- Optional: Phase 5 complete (code quality verified)
- External: `pytest-asyncio`, potentially `httpx-sse` for SSE client

**Checkpoint**: Streaming events verified end-to-end through actual server infrastructure

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - Core streaming utilities
- **User Story 2 (Phase 4)**: Depends on User Story 1 - Integration with existing tool
- **Polish (Phase 5)**: Depends on all user stories being complete
- **Integration Tests (Phase 6)**: Depends on Phase 4 - Validates streaming through actual infrastructure

### User Story Dependencies

- **User Story 1 (US1)**: Can start after Foundational (Phase 2) - No dependencies on US2
- **User Story 2 (US2)**: Depends on User Story 1 completion - Uses streaming utilities

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Type definitions before implementation
- Core functions before integration functions
- Verify tests pass after implementation

### Parallel Opportunities

- **Phase 1**: T003 and T004 can run in parallel with T001/T002
- **Phase 2**: T005 and T006 can run in parallel
- **Phase 3**: T007 and T008 can run in parallel (tests), then sequential implementation
- **Phase 4**: T015 and T016 can run in parallel (tests), then sequential implementation
- **Phase 5**: All code quality tasks (T022-T026) can run in parallel, validation tasks (T027-T029) in parallel
- **Phase 6**: T030 and T031 can run in parallel (setup), then T032-T034 can run in parallel (different test files)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Create test file tests/test_streaming/test_progress.py"
Task: "Create test file tests/test_streaming/test_events.py"

# Then implement sequentially:
Task: "Implement EvaluationProgress dataclass"
Task: "Implement create_adk_progress_event()"
Task: "Implement create_adk_error_event()"
Task: "Implement create_adk_result_event()"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Streaming utilities work independently
5. Other features (004-gym-evaluation) can start using utilities

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 -> Test independently -> Utilities available for 004
3. Add User Story 2 -> Test independently -> RunTau2Evaluation streams progress
4. Polish phase ensures quality and documentation
5. Integration tests (Phase 6) -> Validates streaming works end-to-end through ADK/A2A infrastructure

### Integration with Downstream Features

- **004-gym-evaluation**: Imports `tau2_agent.streaming` utilities for GymOrchestrator
- **007-datadog-project**: Instruments trace context (evaluation_id, task_id, domain) from streaming events

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently testable
- Verify tests fail before implementing
- Commit after each task or logical group
- File structure per plan.md: `tau2_agent/streaming/` for source, `tests/test_streaming/` for tests
- Code quality per CONTRIBUTING.md: Ruff linting (88 char lines), type hints, docstrings on public APIs
