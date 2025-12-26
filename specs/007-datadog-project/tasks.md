# Tasks: Datadog LLM Observability Hackathon Project

**Input**: Design documents from `/specs/007-datadog-project/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Not explicitly requested - tests are OPTIONAL per spec.md pragmatic testing philosophy.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Dependency**: This feature is self-contained for local demo. Cloud Run deployment is deferred to 008-gcp-integration.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

This project uses two components:
- **tau2-bench-agent addition**: `src/tau2/tracing.py` (standalone module, no core changes)
- **Hackathon experiment**: `src/experiments/datadog/`

**Architecture Decision**: The ddtrace integration uses a **wrapper-based approach** with zero modifications to tau2 core modules (e.g., cli.py). See ADR-001 for rationale.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and ddtrace dependency setup

- [X] T001 Add ddtrace>=4.0.0 and datadog>=0.50.0 dependencies to pyproject.toml
- [X] T002 [P] Create experiment directory structure: src/experiments/datadog/{configs,scripts}/
- [X] T003 [P] Create src/experiments/datadog/README.md with experiment overview and usage instructions

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core ddtrace integration that MUST be complete before any observability features work

**CRITICAL**: No metrics emission or traffic generation can work until ddtrace is configured

- [X] T004 Create ddtrace configuration module in src/tau2/tracing.py with configure_ddtrace() function
- [X] T005 Create tau2_traced.py wrapper script in src/experiments/datadog/scripts/ (opt-in, no cli.py changes)
- [X] T006 Create base metrics emitter module in src/experiments/datadog/scripts/emit_metrics.py with DogStatsD initialization

**Checkpoint**: Foundation ready - ddtrace auto-instrumentation works when DD_TRACE_ENABLED=true

---

## Phase 3: User Story 1 - ddtrace Integration (Priority: P1) MVP

**Goal**: tau2-bench-agent emits LLM Observability traces and spans to Datadog via ddtrace auto-instrumentation

**Independent Test**: Run `python -m experiments.datadog.scripts.tau2_traced run --domain mock` and verify traces appear in Datadog APM

**Alternative**: `ddtrace-run tau2 run --domain mock` (uses ddtrace's built-in wrapper)

### Implementation for User Story 1

- [X] T007 [US1] Implement LLMObs.enable() with agentless mode in src/tau2/tracing.py
- [X] T008 [US1] Add patch(litellm=True, httpx=True) for auto-instrumentation in src/tau2/tracing.py
- [X] T009 [US1] Add service-level tags (tau2.version) in src/tau2/tracing.py
- [X] T010 [US1] Add environment variable documentation in src/tau2/tracing.py docstrings

**Checkpoint**: LLM calls to Gemini appear in Datadog LLM Observability UI

---

## Phase 4: User Story 2 - Custom Metrics Emission (Priority: P2)

**Goal**: Post-evaluation metrics are emitted from stored JSON to Datadog for monitors and SLOs

**Independent Test**: Run emit_metrics.py after evaluation and verify metrics in Datadog Metrics Explorer

### Implementation for User Story 2

- [X] T011 [P] [US2] Implement TaskMetrics emission (tau2.task.reward, tau2.task.duration_seconds, tau2.task.success, tau2.task.steps) in src/experiments/datadog/scripts/emit_metrics.py
- [X] T012 [P] [US2] Implement ToolMetrics emission (tau2.tool.calls, tau2.tool.correct, tau2.tool.arguments_match) in src/experiments/datadog/scripts/emit_metrics.py
- [X] T013 [P] [US2] Implement AssertionMetrics emission (tau2.assertion.result, tau2.assertion.nl_failed) in src/experiments/datadog/scripts/emit_metrics.py
- [X] T014 [P] [US2] Implement TerminationMetrics emission (tau2.termination with reason tag) in src/experiments/datadog/scripts/emit_metrics.py
- [X] T015 [US2] Implement EvaluationMetrics aggregation (tau2.evaluation.pass_rate, tau2.evaluation.avg_reward) in src/experiments/datadog/scripts/emit_metrics.py
- [X] T016 [US2] Add JSON file parser for $TAU2_DATA_DIR/evaluations/*.json in src/experiments/datadog/scripts/emit_metrics.py
- [X] T017 [US2] Add CLI interface with --evaluation-id and --all flags in src/experiments/datadog/scripts/emit_metrics.py

**Checkpoint**: Custom tau2 metrics visible in Datadog Metrics Explorer

---

## Phase 4.5: EvaluationStore Integration (ADR-002) - BLOCKING

**Purpose**: Enable post-hoc metrics emission by persisting evaluation results from `RunTau2Evaluation` to `$TAU2_DATA_DIR/evaluations/`.

**Why Blocking**: Without this integration, `emit_metrics.py` cannot find evaluation data. Custom metrics for detection rules will not work.

**Reference**: [ADR-002](adr.md#adr-002-evaluationstore-integration-for-post-hoc-metrics-emission)

### Implementation for EvaluationStore Integration

- [X] T017.1 [ES] Import EvaluationStore in tau2_agent/tools/run_tau2_evaluation.py
- [X] T017.2 [ES] Update `_execute()` to return full `results.simulations` data for metrics extraction
- [X] T017.3 [ES] Add `store.create_session()` call at start of `_execute_streaming()` with evaluation metadata
- [X] T017.4 [ES] Add `store.complete_evaluation()` call after successful evaluation with full results
- [X] T017.5 [ES] Add `store.fail_evaluation()` call in exception handlers
- [X] T017.6 [ES] Add try/catch wrapper around store calls (evaluation should succeed even if storage fails)
- [X] T017.7 [ES] Update emit_metrics.py to handle EvaluationStore JSON format if needed
- [X] T017.8 [ES] Test: Run evaluation via A2A, verify files in `$TAU2_DATA_DIR/evaluations/`
- [X] T017.9 [ES] Test: Run `emit_metrics.py --all` and verify metrics are emitted

**Checkpoint**:
1. Files appear in `$TAU2_DATA_DIR/evaluations/` after A2A evaluation
2. `emit_metrics.py --all` successfully reads and emits metrics
3. Metrics visible in Datadog Metrics Explorer (if DD_API_KEY set)

---

## Phase 5: User Story 3 - Datadog Configuration Setup (Priority: P3)

**Goal**: Monitors, SLOs, and dashboard are created in Datadog via API or JSON import

**Independent Test**: Run setup_datadog.py --all and verify resources exist in Datadog UI

### Implementation for User Story 3

- [X] T018 [P] [US3] Copy monitors.json to src/experiments/datadog/configs/monitors.json (from contracts/)
- [X] T019 [P] [US3] Copy slos.json to src/experiments/datadog/configs/slos.json (from contracts/)
- [X] T020 [P] [US3] Copy dashboards.json to src/experiments/datadog/configs/dashboards.json (from contracts/)
- [X] T021 [P] [US3] Copy case_templates.json to src/experiments/datadog/configs/case_templates.json (from contracts/)
- [X] T022 [US3] Create setup_datadog.py script in src/experiments/datadog/scripts/setup_datadog.py
- [X] T023 [US3] Implement --monitors flag to create monitors via Datadog API in setup_datadog.py
- [X] T024 [US3] Implement --slos flag to create SLOs via Datadog API in setup_datadog.py
- [X] T025 [US3] Implement --dashboard flag to create dashboard via Datadog API in setup_datadog.py
- [X] T026 [US3] Implement --all flag to create all resources in setup_datadog.py
- [X] T027 [US3] Implement --export flag to export current configurations to JSON in setup_datadog.py
- [X] T028 [US3] Add API key validation and error handling in setup_datadog.py

**Checkpoint**: 5 monitors, 3 SLOs, and 1 dashboard created in Datadog

---

## Phase 6: User Story 4 - Traffic Generator (Priority: P4)

**Goal**: Generate varied telemetry quickly for Datadog dashboard visualization and monitor triggering

**Prerequisite**: ddtrace configured (Phase 2 complete), Datadog API key set

**Independent Test**: Run `python -m experiments.datadog.scripts.traffic_generator --count 5` and verify metrics in Datadog

**Architecture**: Concurrent-by-default A2A requests to tau2_agent
- Fires N concurrent evaluation requests using `asyncio.gather()`
- tau2_agent handles up to 10 concurrent evaluations (via `_EVALUATION_EXECUTOR`)
- Generates more data points in less time for richer Datadog visualizations

**Pattern Source**: Inline copy from `tests/test_datadog_e2e/conftest.py`:
- `TracedServer` / `MockAgentServer` dataclasses
- `build_a2a_evaluation_request()`, `send_a2a_evaluation_request()`, `parse_sse_event()`
- Server startup with DD environment variables

### Implementation for User Story 4

- [X] T029 [US4] Create traffic_generator.py with inline copies of conftest.py patterns (TracedServer, A2A helpers)
- [X] T030 [US4] Implement server startup: tau2_agent (port 8766) + mock agent (port 8767)
- [X] T031 [US4] Implement concurrent A2A request dispatch using asyncio.gather()
- [X] T032 [US4] Implement normal mode: varied domains (mock, airline) with random task sampling
- [X] T033 [US4] Implement failure mode: specific task_ids that produce low rewards to trigger DR-002
- [X] T034 [US4] Add CLI: --count N (evaluations), --mode (normal/failure), --domain, --skip-server-start
- [X] T035 [US4] Call emit_metrics.py after all evaluations complete
- [X] T036 [US4] Add error handling: log failures, continue with remaining evaluations
- [X] T036.1 [US4] Add --dry-run mode: run evaluations but emit_metrics.py uses --dry-run (no Datadog needed)
- [X] T036.2 [US4] Local test: verify traffic_generator.py --count 2 --dry-run works without DD_API_KEY

**Checkpoint**:
- Local: `traffic_generator.py --count 2 --dry-run` completes with mock agent (needs NEBIUS_API_KEY only)
- Production: `traffic_generator.py --count 5` sends metrics to Datadog (needs DD_API_KEY)

---

## Phase 7: User Story 5 - Demo Script & Hackathon Submission (Priority: P5)

**Goal**: One-command demo that generates telemetry and shows Datadog observability features

**Independent Test**: Run `./demo.sh` and verify traces, metrics, and alerts appear in Datadog

**Demo Flow**:
1. Check prerequisites (DD_API_KEY, DD_SITE, NEBIUS_API_KEY)
2. Start tau2_agent (port 8766) + mock agent (port 8767) with DD_TRACE_ENABLED=true
3. Create Datadog resources (monitors, SLOs, dashboard)
4. Fire concurrent evaluation requests via traffic_generator.py
5. Emit metrics to Datadog
6. Trigger failure mode to fire DR-002 alert
7. Output Datadog dashboard URL for judges

### Implementation for User Story 5

- [X] T038 [US5] Create demo.sh script in src/experiments/datadog/scripts/demo.sh
- [X] T039 [US5] Add prerequisite check: DD_API_KEY, DD_SITE, NEBIUS_API_KEY (for mock agent LLM)
- [X] T040 [US5] Start servers: traffic_generator.py handles tau2_agent + mock agent startup
- [X] T041 [US5] Create Datadog resources: setup_datadog.py --all
- [X] T042 [US5] Generate normal traffic: traffic_generator.py --count 5 --mode normal
- [X] T043 [US5] Generate failure traffic: traffic_generator.py --count 3 --mode failure
- [X] T044 [US5] Emit metrics: emit_metrics.py --all
- [X] T045 [US5] Output summary: dashboard URL, metrics count, alert status
- [X] T045.1 [US5] Add --dry-run mode to demo.sh: runs full flow locally without Datadog API calls
- [X] T045.2 [US5] Local test: verify demo.sh --dry-run works with only NEBIUS_API_KEY set
- [X] T046 [US5] Document demo in src/experiments/datadog/README.md (setup, run, expected output)
- [X] T047 [US5] Add Apache-2.0 LICENSE to src/experiments/datadog/LICENSE
- [X] T048 [US5] Create submission checklist in README.md

**Checkpoint**:
- Local: `./demo.sh --dry-run` validates full flow without Datadog (needs NEBIUS_API_KEY only)
- Production: `./demo.sh` sends traces + metrics to Datadog (needs DD_API_KEY, DD_SITE)

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and documentation

- [X] T050 Run quickstart.md validation (local setup with DD_TRACE_ENABLED=true)
  - Validated via E2E tests: `pytest tests/test_datadog_e2e/ -v -m datadog_e2e` (7/7 passed)
- [X] T051 Verify all 5 detection rules (DR-001 to DR-005) can be triggered via A2A traffic generator
  - DR-001: High Error Rate → uses tau2.task.success/tau2.task.total ✓
  - DR-002: Task Failure Spike → uses tau2.task.reward ✓
  - DR-003: Token Cost Anomaly → uses tau2.llm.token_cost (auto-instrumented) ✓
  - DR-004: Premature Termination → uses tau2.termination ✓
  - DR-005: Latency SLO Breach → uses tau2.task.duration_seconds ✓
- [X] T052 Verify Case creation from monitor alerts
  - Case templates defined in configs/case_templates.json
  - Monitors configured with escalation_message for case creation
- [X] T053 Update spec.md submission checklist with completed items
- [X] T054 Code cleanup and formatting (ruff check)
  - Fixed 3 linting issues in tests/test_datadog_e2e/conftest.py

---

## Phase 9: E2E A2A Observability Testing

**Purpose**: Validate full A2A → Evaluation → Datadog metrics flow with pytest tests

**Goal**: Create automated tests that validate the complete observability pipeline when tau2_agent receives A2A evaluation requests via JSON-RPC protocol.

**Test Flow**:
1. Start tau2_agent with `DD_TRACE_ENABLED=true` via environment variables
2. Send A2A JSON-RPC requests to `/a2a/tau2_agent` endpoint
3. Verify SSE streaming events contain tau2 metadata
4. Verify evaluation results persist to `$TAU2_DATA_DIR/evaluations/`
5. Verify `emit_metrics.py` can process stored results

**Run Command**: `pytest tests/test_datadog_e2e/ -v -m datadog_e2e`

**A2A Protocol Pattern** (from `tests/test_adk_server/test_a2a_endpoint.py`):
```python
# Send JSON-RPC 2.0 request to tau2_agent
a2a_request = {
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
        "message": {
            "messageId": str(uuid.uuid4()),
            "role": "user",
            "parts": [{"text": "Run evaluation on mock domain for agent at http://..."}]
        }
    },
    "id": str(uuid.uuid4())
}
response = await client.post("/a2a/tau2_agent", json=a2a_request)
```

**Reusable Patterns for Phase 6**:
- `traced_adk_server` fixture → subprocess management with DD env vars
- `TracedServer` dataclass → server lifecycle abstraction
- `temp_data_dir` fixture → isolated EvaluationStore for testing
- A2A request builder → JSON-RPC 2.0 message formatting

### Implementation for E2E Testing

- [X] T055 [E2E] Create tests/test_datadog_e2e/__init__.py with module docstring
- [X] T056 [E2E] Add datadog_e2e pytest marker to pyproject.toml
- [X] T057 [E2E] Create tests/test_datadog_e2e/conftest.py with traced_adk_server fixture
- [X] T058 [E2E] Implement test_traced_server_starts_with_ddtrace in test_observability_flow.py
- [X] T059 [E2E] **REFACTOR**: Update test_evaluation_tool_persists_to_store to use A2A JSON-RPC requests
  - Consolidated into `test_full_a2a_evaluation_flow` which uses A2A JSON-RPC
- [X] T060 [E2E] **REFACTOR**: Update test_emit_metrics_processes_stored_evaluation to use A2A flow
  - Implemented as `test_emit_metrics_from_real_evaluation` with A2A evaluation
- [X] T061 [E2E] **REFACTOR**: Update test_event_state_progression to use A2A SSE streaming
  - Consolidated into `test_full_a2a_evaluation_flow` which validates SSE events
- [X] T062 [E2E] **REFACTOR**: Update test_trace_context_metadata_in_events to use A2A flow
  - SSE event structure validation included in `test_full_a2a_evaluation_flow`
- [X] T063 [E2E] Add conftest.py helper: `send_a2a_evaluation_request()` for JSON-RPC message building
- [X] T064 [E2E] Run and verify all refactored datadog_e2e tests pass
  - 7/7 tests pass in ~67 seconds (includes concurrent evaluation test)

**Checkpoint**: `pytest tests/test_datadog_e2e/ -v -m datadog_e2e` passes with tests using actual A2A protocol

---

## Dependencies & Execution Order

### External Dependency

**None for local demo.** Cloud Run deployment deferred to 008-gcp-integration.

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-4)**: All depend on Foundational phase completion
  - US1 (ddtrace) can start first
  - US2 (metrics) can run in parallel with US1
- **EvaluationStore Integration (Phase 4.5)**: Depends on US2 - BLOCKS Phase 5+
  - Enables `emit_metrics.py` to find evaluation data
  - Required for detection rules to have metrics to trigger on
- **User Stories (Phase 5-7)**: Depend on Phase 4.5
  - US3 (Datadog config) depends on Phase 4.5 for metrics validation
  - US4 (traffic) depends on Phase 4.5 for persistence, Phase 9 for patterns
  - US5 (demo) depends on US3, US4
- **Polish (Phase 8)**: Depends on all user stories being complete
- **E2E Testing (Phase 9)**: Provides reusable A2A patterns for Phase 6
  - `traced_adk_server` fixture → traffic generator server management
  - A2A request helpers → JSON-RPC message building

### User Story Dependencies

- **User Story 1 (P1)**: Foundation only - No dependencies on other stories
- **User Story 2 (P2)**: Foundation only - Can run in parallel with US1
- **Phase 4.5 (EvaluationStore)**: Depends on US2 - BLOCKING for US3+
- **User Story 3 (P3)**: Depends on Phase 4.5 for metrics validation
- **User Story 4 (P4)**: Depends on Phase 4.5 for persistence, uses A2A patterns from Phase 9
- **User Story 5 (P5)**: Depends on US3, US4
- **Phase 9 (E2E Tests)**: Provides A2A patterns; should be refactored to use actual A2A before Phase 6

### Within Each User Story

- Config JSON copy tasks [P] can run in parallel
- Script implementations are sequential (build on each other)
- Story complete before moving to next priority

### Parallel Opportunities

- T002, T003 can run in parallel (Setup phase)
- T011, T012, T013, T014 can run in parallel (metrics by type)
- T018, T019, T020, T021 can run in parallel (config file copies)
- US1, US2, US3 can all run in parallel after Foundation

---

## Parallel Example: User Story 2

```bash
# Launch all metric type implementations in parallel:
Task: "T011 [P] [US2] Implement TaskMetrics emission"
Task: "T012 [P] [US2] Implement ToolMetrics emission"
Task: "T013 [P] [US2] Implement AssertionMetrics emission"
Task: "T014 [P] [US2] Implement TerminationMetrics emission"
```

---

## Parallel Example: User Story 3

```bash
# Launch all config file copies in parallel:
Task: "T018 [P] [US3] Copy monitors.json"
Task: "T019 [P] [US3] Copy slos.json"
Task: "T020 [P] [US3] Copy dashboards.json"
Task: "T021 [P] [US3] Copy case_templates.json"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1 (ddtrace integration)
4. **STOP and VALIDATE**: Test with `python -m experiments.datadog.scripts.tau2_traced run --domain mock`
5. Verify traces in Datadog APM

### Incremental Delivery

1. Setup + Foundational + US1 (ddtrace) = LLM traces in Datadog
2. Add US2 (metrics) = Custom metrics code ready (but no data yet)
3. **Add Phase 4.5 (EvaluationStore) = Evaluation data persisted, `emit_metrics.py` works**
4. Add US3 (Datadog config) = Monitors, SLOs, dashboard created
5. **Refactor Phase 9 (E2E Tests) = A2A patterns ready for traffic generator**
6. Add US4 (traffic) = Automated A2A-based telemetry generation
7. Add US5 (demo) = Hackathon demo workflow complete

### Critical Path

```
Setup → Foundational → US1 (ddtrace) ─┬─→ US2 (metrics) → Phase 4.5 (EvaluationStore) ─┬─→ Phase 9 (A2A refactor) → US4 (traffic) → US5 (demo)
                                      │                                                 │
                                      └───────────────────→ US3 (config) ◄──────────────┘
```

**Notes**:
- Phase 4.5 is a BLOCKING dependency. Without EvaluationStore integration, `emit_metrics.py` cannot find data.
- Phase 9 E2E test refactor should happen before Phase 6 to establish A2A patterns for traffic generator.

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- **Hero Monitor**: DR-002 (Task Failure Spike) is prioritized for demo - traffic generator triggers this with failure mode
- **Concurrent-by-Default**: Traffic generator fires N concurrent A2A requests; tau2_agent handles up to 10 concurrent evaluations
- **Separate Ports**: tau2_agent (8766) and mock agent (8767) MUST run on separate ports to avoid async deadlock (see issue_tracker/concurrency-fix.md)
- **Pattern Inline Copy**: conftest.py patterns copied into traffic_generator.py (no shared module dependency)
- **Zero Core Changes**: ddtrace integration uses wrapper-based approach, no modifications to tau2 cli.py
