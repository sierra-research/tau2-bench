# Tasks: Evaluation Store

**Input**: Design documents from `/specs/002-evaluation-store/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are included per plan.md constitution check (Testing Philosophy section).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/tau2/store/` module alongside existing tau2 modules
- **Tests**: `tests/test_store/` mirroring source structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create module structure and copy contract files

- [X] T001 Create src/tau2/store/ directory structure
- [X] T002 [P] Create src/tau2/store/__init__.py with public API exports
- [X] T003 [P] Create src/tau2/store/exceptions.py from contracts/exceptions.py
- [X] T004 [P] Create src/tau2/store/models.py from contracts/models.py
- [X] T005 Create tests/test_store/ directory structure

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core utilities and configuration that all user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Implement configuration loader in src/tau2/store/config.py (TAU2_DATA_DIR, retention settings, log settings)
- [X] T007 Implement atomic_write utility function in src/tau2/store/utils.py (temp file + rename pattern)
- [X] T008 [P] Implement generate_evaluation_id function in src/tau2/store/utils.py (eval-{unix_ms}-{random_6_chars})
- [X] T009 Implement directory initialization in src/tau2/store/utils.py (ensure sessions/, evaluations/, logs/ exist)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Core Storage (Priority: P0)

**Goal**: Save and retrieve evaluations with atomic writes, session management, two-directory design

**Independent Test**: Can create a session, update progress, complete evaluation, and retrieve it by ID

### Tests for User Story 1

- [X] T010 [P] [US1] Create tests/test_store/test_store.py with test fixtures
- [X] T011 [P] [US1] Write test_create_session in tests/test_store/test_store.py
- [X] T012 [P] [US1] Write test_update_progress in tests/test_store/test_store.py
- [X] T013 [P] [US1] Write test_complete_evaluation in tests/test_store/test_store.py
- [X] T014 [P] [US1] Write test_fail_evaluation in tests/test_store/test_store.py
- [X] T015 [P] [US1] Write test_get_evaluation_from_session in tests/test_store/test_store.py
- [X] T016 [P] [US1] Write test_get_evaluation_from_completed in tests/test_store/test_store.py
- [X] T017 [P] [US1] Write test_list_evaluations in tests/test_store/test_store.py
- [X] T018 [P] [US1] Write test_evaluation_id_collision in tests/test_store/test_store.py

### Implementation for User Story 1

- [X] T019 [US1] Implement EvaluationStore class skeleton in src/tau2/store/store.py
- [X] T020 [US1] Implement create_session method in src/tau2/store/store.py
- [X] T021 [US1] Implement update_progress method in src/tau2/store/store.py
- [X] T022 [US1] Implement complete_evaluation method in src/tau2/store/store.py (moves session to evaluations/)
- [X] T023 [US1] Implement fail_evaluation method in src/tau2/store/store.py
- [X] T024 [US1] Implement get_evaluation method in src/tau2/store/store.py (checks sessions first, then evaluations)
- [X] T025 [US1] Implement list_evaluations method in src/tau2/store/store.py (filters by domain, status)
- [X] T026 [US1] Implement create_store factory function in src/tau2/store/store.py
- [X] T027 [US1] Export create_store in src/tau2/store/__init__.py

**Checkpoint**: User Story 1 complete - can create, track, complete, and retrieve evaluations

---

## Phase 4: User Story 2 - Observability Integration (Priority: P0)

**Goal**: Include trace_id, session_id, state_history for OTel/Datadog correlation and SSE reconnection

**Independent Test**: Can look up an evaluation by trace_id, state_history records all transitions

### Tests for User Story 2

- [X] T028 [P] [US2] Write test_get_evaluation_by_trace_id in tests/test_store/test_store.py
- [X] T029 [P] [US2] Write test_state_history_transitions in tests/test_store/test_store.py
- [X] T030 [P] [US2] Write test_session_heartbeat_updates in tests/test_store/test_store.py

### Implementation for User Story 2

- [X] T031 [US2] Implement get_evaluation_by_trace_id method in src/tau2/store/store.py
- [X] T032 [US2] Add state_history tracking to create_session in src/tau2/store/store.py
- [X] T033 [US2] Add state_history transitions to update_progress in src/tau2/store/store.py
- [X] T034 [US2] Add state_history transitions to complete_evaluation and fail_evaluation in src/tau2/store/store.py
- [X] T035 [US2] Export get_evaluation_by_trace_id via EvaluationStore protocol

**Checkpoint**: User Story 2 complete - evaluations have full observability context

---

## Phase 5: User Story 3 - Retention & Cleanup (Priority: P1)

**Goal**: Automatic deletion of expired evaluations, stale session detection, abandoned session cleanup

**Independent Test**: Can mark stale sessions as abandoned, can delete expired evaluations by file age

### Tests for User Story 3

- [X] T036 [P] [US3] Create tests/test_store/test_retention.py with test fixtures
- [X] T037 [P] [US3] Write test_cleanup_expired_evaluations in tests/test_store/test_retention.py
- [X] T038 [P] [US3] Write test_cleanup_failed_evaluations_shorter_retention in tests/test_store/test_retention.py
- [X] T039 [P] [US3] Write test_mark_abandoned_sessions in tests/test_store/test_retention.py
- [X] T040 [P] [US3] Write test_cleanup_abandoned_sessions in tests/test_store/test_retention.py
- [X] T041 [P] [US3] Write test_file_age_cleanup in tests/test_store/test_retention.py

### Implementation for User Story 3

- [X] T042 [US3] Create src/tau2/store/retention.py with RetentionManager class skeleton
- [X] T043 [US3] Implement cleanup_expired_evaluations in src/tau2/store/retention.py (file-age based)
- [X] T044 [US3] Implement mark_abandoned_sessions in src/tau2/store/retention.py (heartbeat check)
- [X] T045 [US3] Implement cleanup_abandoned_sessions in src/tau2/store/retention.py
- [X] T046 [US3] Implement create_retention_manager factory function in src/tau2/store/retention.py
- [X] T047 [US3] Export RetentionManager and create_retention_manager in src/tau2/store/__init__.py

**Checkpoint**: User Story 3 complete - automatic retention and cleanup works

---

## Phase 6: User Story 4 - Structured Logging (Priority: P1)

**Goal**: JSON Lines event logging with rotation, compression, and stdout output

**Independent Test**: Can emit events to events.jsonl, logs contain trace_id, stdout output works

### Tests for User Story 4

- [X] T048 [P] [US4] Create tests/test_store/test_events.py with test fixtures
- [X] T049 [P] [US4] Write test_log_event_to_file in tests/test_store/test_events.py
- [X] T050 [P] [US4] Write test_log_event_with_trace_id in tests/test_store/test_events.py
- [X] T051 [P] [US4] Write test_log_event_to_stdout in tests/test_store/test_events.py
- [X] T052 [P] [US4] Write test_standard_event_types in tests/test_store/test_events.py

### Implementation for User Story 4

- [X] T053 [US4] Create src/tau2/store/events.py with EventLogger class skeleton
- [X] T054 [US4] Implement log_event method in src/tau2/store/events.py (JSON Lines output)
- [X] T055 [US4] Add stdout output support to EventLogger in src/tau2/store/events.py (TAU2_LOG_STDOUT)
- [X] T056 [US4] Implement rotate_logs in src/tau2/store/retention.py (compression, deletion)
- [X] T057 [US4] Implement create_event_logger factory function in src/tau2/store/events.py
- [X] T058 [US4] Export EventLogger and create_event_logger in src/tau2/store/__init__.py

**Checkpoint**: User Story 4 complete - structured logging with correlation IDs works

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, integration, and cleanup

- [X] T059 Verify all tests pass with pytest tests/test_store/
- [X] T060 Add type hints validation (run mypy on src/tau2/store/)
- [X] T061 Add Google-style docstrings to all public functions
- [X] T062 Run quickstart.md validation scenarios manually
- [X] T063 Verify file permissions (0o600 for data, 0o640 for logs)
- [X] T064 Update src/tau2/store/__init__.py with complete public API exports
- [X] T065 Verify exception exports (EvaluationNotFoundError, InvalidStateError, EvaluationIdCollisionError)

---

## Dependencies & Execution Order

### Important
- **Use uv**: ensure we use uv to run python or pip commands so that it automatically sources the venv
- **Quality checks**: run ruff and mypy checks on recently implemented code

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1 (Core Storage) and US2 (Observability) are P0 - do first
  - US3 (Retention) and US4 (Logging) are P1 - do after P0 stories
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P0)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P0)**: Can start after Foundational - Builds on US1's store methods but independently testable
- **User Story 3 (P1)**: Can start after Foundational - Uses store for reading, independently testable
- **User Story 4 (P1)**: Can start after Foundational - Independent logging module

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Implementation follows contracts/ definitions
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel (T002, T003, T004)
- All Foundational config tasks can run sequentially (T006-T009)
- All tests within a user story marked [P] can run in parallel
- Once Foundational phase completes, US1 and US2 can start in parallel (both P0)
- Once US1 and US2 complete, US3 and US4 can start in parallel (both P1)

---

## Parallel Example: User Story 1 Tests

```bash
# Launch all tests for User Story 1 together:
Task: "Create tests/test_store/test_store.py with test fixtures"
Task: "Write test_create_session in tests/test_store/test_store.py"
Task: "Write test_update_progress in tests/test_store/test_store.py"
Task: "Write test_complete_evaluation in tests/test_store/test_store.py"
Task: "Write test_fail_evaluation in tests/test_store/test_store.py"
Task: "Write test_get_evaluation_from_session in tests/test_store/test_store.py"
Task: "Write test_get_evaluation_from_completed in tests/test_store/test_store.py"
Task: "Write test_list_evaluations in tests/test_store/test_store.py"
Task: "Write test_evaluation_id_collision in tests/test_store/test_store.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Core Storage)
4. Complete Phase 4: User Story 2 (Observability)
5. **STOP and VALIDATE**: Test US1 + US2 independently
6. Deploy/demo if ready - evaluations can be saved, retrieved, and correlated with traces

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 (Core Storage) -> Test independently -> MVP core functionality
3. Add User Story 2 (Observability) -> Test independently -> OTel/Datadog ready
4. Add User Story 3 (Retention) -> Test independently -> Production cleanup ready
5. Add User Story 4 (Logging) -> Test independently -> Full observability
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Core Storage)
   - Developer B: User Story 2 (Observability)
3. Once P0 stories done:
   - Developer A: User Story 3 (Retention)
   - Developer B: User Story 4 (Logging)
4. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Implementation follows contracts/ definitions exactly
- All timestamps use UTC with Z suffix per spec
- File permissions: 0o600 for data files, 0o640 for log files
