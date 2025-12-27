# Tasks: 008-GCP Integration

**Input**: Design documents from `/specs/008-gcp-integration/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/byok-headers.yaml

**Tests**: Test tasks included as specified in plan.md Testing section.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md structure - existing `tau2_agent/` directory extension with tests in `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency configuration

- [X] T001 [P] Add google-cloud-secret-manager>=2.18.0 to tau2_agent/docker_setup/requirements.txt
- [X] T002 [P] Create tau2_agent/config.py with EvaluationLimits constants (MAX_TASKS=30, MAX_TRIALS=3, TIMEOUT_SECONDS=3600)
- [X] T003 [P] Create tau2_agent/scripts/setup-secrets.sh for GCP Secret Manager setup (create google-api-key secret, grant IAM access)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Create tau2_agent/context.py with BYOKContext dataclass and contextvars (user_llm_model, user_llm_api_key, request_id)
- [X] T004.1 [P] Create tau2_agent/errors.py with ErrorCode enum (MISSING_HEADER, INVALID_AUTH, USER_LLM_AUTH_FAILED, LIMIT_EXCEEDED, EVALUATION_FAILED) with docstring describing each code
- [X] T004.2 [P] Add EvaluationError dataclass to tau2_agent/errors.py (code: ErrorCode, message: str, details: dict[str, Any] | None) with to_dict() method
- [X] T004.3 [P] Create tests/test_tau2_agent/test_errors.py with unit tests for EvaluationError.to_dict() output format
- [X] T005 Create tau2_agent/server.py entrypoint that imports get_fast_api_app() from ADK and runs uvicorn with PORT env var support
- [X] T006 Create tests/test_tau2_agent/test_context.py with unit tests for context variable set/get/reset lifecycle

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - BYOK Header Extraction (Priority: P1) 🎯 MVP

**Goal**: Clients can provide LLM credentials via X-User-LLM-Model and X-User-LLM-API-Key HTTP headers

**Independent Test**: Send request with BYOK headers → headers are extracted and available in request context

### Tests for User Story 1

- [X] T007 [P] [US1] Create tests/test_tau2_agent/test_middleware.py with unit tests for BYOKMiddleware header extraction
- [X] T008 [P] [US1] Add tests for missing header 400 responses in tests/test_tau2_agent/test_middleware.py

### Implementation for User Story 1

- [X] T009 [US1] Create tau2_agent/middleware.py with BYOKMiddleware class inheriting BaseHTTPMiddleware
- [X] T010 [US1] Implement header extraction (x-user-llm-model, x-user-llm-api-key) with case-insensitive access in tau2_agent/middleware.py
- [X] T011 [US1] Add 400 Bad Request response using EvaluationError(MISSING_HEADER) for missing required headers in tau2_agent/middleware.py
- [X] T012 [US1] Implement context variable set/reset in middleware dispatch method (token pattern with finally block) in tau2_agent/middleware.py
- [X] T013 [US1] Register BYOKMiddleware in tau2_agent/server.py via app.add_middleware()
- [X] T014 [US1] Add loguru logging for header validation (never log API keys) in tau2_agent/middleware.py

**Checkpoint**: At this point, BYOK headers are extracted and stored in request context

---

## Phase 4: User Story 2 - Server-Configured Orchestrator LLM (Priority: P1)

**Goal**: tau2_agent uses server-configured Gemini model for orchestration without client input

**Independent Test**: Agent starts with TAU2_AGENT_MODEL env var and uses GOOGLE_API_KEY for Gemini API

### Implementation for User Story 2

- [X] T015 [P] [US2] Add ServerConfig dataclass to tau2_agent/config.py (tau2_agent_model, google_api_key, port, log_level, service_api_keys)
- [X] T016 [US2] Modify tau2_agent/agent.py create_model() function to read TAU2_AGENT_MODEL env var with gemini-2.0-flash default
- [X] T017 [US2] Remove Nebius-specific logic from tau2_agent/agent.py (simplify for GCP deployment)
- [X] T018 [US2] Ensure GOOGLE_API_KEY environment variable is used by litellm for Gemini API in tau2_agent/agent.py

**Checkpoint**: Orchestrator LLM configured from server environment

---

## Phase 5: User Story 3 - Request Validation & Limit Enforcement (Priority: P1)

**Goal**: Validate incoming requests and enforce task/trial limits for Cloud Run timeout compliance

**Independent Test**: Request with num_tasks > 30 returns 400 error with explanation

### Tests for User Story 3

- [X] T019 [P] [US3] Add validation tests to tests/test_tau2_agent/test_middleware.py for limit enforcement

### Implementation for User Story 3

- [X] T020 [US3] Modify tau2_agent/tools/run_tau2_evaluation.py to import and validate against MAX_TASKS and MAX_TRIALS from config.py
- [X] T021 [US3] Raise EvaluationError(LIMIT_EXCEEDED) with details dict when num_tasks > 30 or num_trials > 3 in run_tau2_evaluation.py
- [X] T022 [US3] Modify tau2_agent/tools/run_tau2_evaluation.py to read user LLM credentials from contextvars instead of os.getenv()
- [X] T023 [US3] Pass user_llm_api_key to RunConfig.llm_args_user in run_tau2_evaluation.py
- [X] T023.1 [US3] Add try/except in run_tau2_evaluation.py to catch litellm AuthenticationError and return EvaluationError with USER_LLM_AUTH_FAILED code

**Checkpoint**: Request validation and BYOK credential passing complete

---

## Phase 6: User Story 4 - Optional Service Authentication (Priority: P2)

**Goal**: Optional API key authentication to restrict service access via Authorization header

**Independent Test**: When SERVICE_API_KEYS configured, requests without valid Bearer token return 401

### Implementation for User Story 4

- [X] T024 [P] [US4] Add tests for optional service auth in tests/test_tau2_agent/test_middleware.py
- [X] T025 [US4] Add Authorization header extraction to tau2_agent/middleware.py
- [X] T026 [US4] Implement Bearer token validation against SERVICE_API_KEYS env var (comma-separated list) in tau2_agent/middleware.py
- [X] T027 [US4] Return 401 Unauthorized using EvaluationError(INVALID_AUTH) when auth enabled but invalid/missing token in tau2_agent/middleware.py

**Checkpoint**: Optional service authentication functional

---

## Phase 7: User Story 5 - Cloud Run Deployment (Priority: P1)

**Goal**: Deploy tau2_agent to Cloud Run with proper configuration

**Independent Test**: curl to deployed service URL returns valid A2A response with BYOK headers

### Implementation for User Story 5

- [X] T028 [P] [US5] Modify tau2_agent/docker_setup/Dockerfile to add ENV PORT=8001, TAU2_AGENT_MODEL=gemini-2.0-flash
- [X] T029 [P] [US5] Update Dockerfile CMD to use python -m tau2_agent.server instead of adk api_server
- [X] T030 [P] [US5] Create tau2_agent/docker_setup/service.yaml with Cloud Run configuration (2Gi memory, 2 CPU, 60min timeout, 0-10 instances)
- [X] T031 [US5] Create tau2_agent/scripts/deploy.sh with gcloud run deploy command including --set-secrets for GOOGLE_API_KEY
- [X] T032 [US5] Create tau2_agent/scripts/test-deployment.sh to verify deployment with sample curl requests

**Checkpoint**: Cloud Run deployment configured and scriptable

---

## Phase 8: Integration Testing

**Goal**: End-to-end BYOK flow validation

### Implementation

- [X] T033 [P] Create tests/integration/test_byok_flow.py with integration tests for full BYOK request flow (mock HTTP)
- [X] T034 Add test for LLM auth failure (401) response in tests/integration/test_byok_flow.py
- [X] T035 Add test for successful evaluation with mocked LLM responses in tests/integration/test_byok_flow.py

**Checkpoint**: Integration tests validate complete BYOK flow

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and final cleanup

- [X] T036 [P] Update tau2_agent/README.md with GCP deployment instructions and BYOK usage
- [X] T037 [P] Add docstrings (Google-style) to tau2_agent/middleware.py, tau2_agent/context.py, tau2_agent/server.py
- [X] T038 Run quickstart.md validation - test deployment steps manually
- [X] T039 Ensure all API keys are never logged (audit loguru calls in new modules)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-7)**: All depend on Foundational phase completion
  - US1 (BYOK Headers) can start after Foundational
  - US2 (Server LLM) can start after Foundational, parallel with US1
  - US3 (Validation) depends on US1 (needs context vars)
  - US4 (Service Auth) depends on US1 (extends middleware)
  - US5 (Deployment) depends on US1-US3 for functional code
- **Integration Testing (Phase 8)**: Depends on US1-US3 completion
- **Polish (Phase 9)**: Depends on all user stories being complete

### User Story Dependencies

```
Foundational (Phase 2)
        │
        ├──────────────┬──────────────┐
        ▼              ▼              │
    US1 (BYOK)    US2 (Server)        │
        │              │              │
        ├──────────────┤              │
        ▼              ▼              │
    US3 (Validation)                  │
        │                             │
        ├──────────────┬──────────────┘
        ▼              ▼
    US4 (Auth)    US5 (Deploy)
        │              │
        └──────┬───────┘
               ▼
        Integration Tests
               │
               ▼
            Polish
```

### Within Each User Story

- Tests (where included) should be written FIRST and FAIL before implementation
- Context/config before middleware
- Middleware before tool modifications
- Core implementation before deployment scripts
- Story complete before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**: All tasks (T001-T003) can run in parallel

**Phase 2 (Foundational)**: T004 blocks T005-T006; T005-T006 can run in parallel after T004

**Phase 3-7 (User Stories)**:
- US1 tests (T007-T008) can run in parallel
- US1 and US2 can proceed in parallel after Foundational
- US3 depends on US1 completion
- US4 depends on US1 completion
- US5 deployment tasks (T028-T030) can run in parallel

**Phase 8 (Integration)**: T033 parallel with implementation completion

**Phase 9 (Polish)**: T036-T037 can run in parallel

---

## Parallel Example: Phase 1 Setup

```bash
# Launch all setup tasks together:
Task: "Add google-cloud-secret-manager to requirements.txt"
Task: "Create config.py with EvaluationLimits"
Task: "Create setup-secrets.sh for GCP Secret Manager"
```

## Parallel Example: User Stories 1 & 2

```bash
# After Foundational complete, launch US1 and US2 in parallel:

# US1 tests:
Task: "Create test_middleware.py with BYOKMiddleware tests"
Task: "Add missing header 400 response tests"

# US2 (parallel with US1):
Task: "Add ServerConfig to config.py"
Task: "Modify agent.py for TAU2_AGENT_MODEL"
```

---

## Implementation Strategy

### MVP First (User Stories 1-3 + 5)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (BYOK Headers)
4. Complete Phase 4: User Story 2 (Server LLM)
5. Complete Phase 5: User Story 3 (Validation)
6. Complete Phase 7: User Story 5 (Deployment) - skip optional auth for MVP
7. **STOP and VALIDATE**: Test deployment with quickstart.md steps
8. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. US1 + US2 → Server starts with BYOK middleware → Local test
3. US3 → Validation complete → Local integration test
4. US5 → Deploy to Cloud Run → **MVP COMPLETE**
5. US4 (optional) → Add service auth if needed
6. Polish → Documentation and cleanup

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (BYOK)
   - Developer B: User Story 2 (Server LLM) + User Story 5 (Deployment infra)
3. After US1 complete:
   - Developer A: User Story 3 (Validation) + User Story 4 (Auth)
   - Developer B: Integration tests
4. Final: Polish phase together

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- **Security**: Never log API keys (user_llm_api_key, google_api_key)
- **ADR Compliance**: All changes isolated to tau2_agent/ per ADR-006
