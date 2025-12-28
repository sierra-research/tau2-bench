# Implementation Plan: 008-GCP Integration

**Branch**: `008-gcp-integration` | **Date**: 2025-12-26 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/008-gcp-integration/spec.md`
**Reference**: [adr.md](./adr.md) (8 ADRs) | [gcp-integration-guide.md](./gcp-integration-guide.md)

## Summary

Deploy tau2_agent as a hosted evaluation service on Google Cloud Platform using Cloud Run. Implement BYOK (Bring Your Own Key) pattern for user simulator LLM credentials via HTTP headers while using server-configured Gemini for orchestration. Enforce task limits (max 30 tasks, max 3 trials) to stay within Cloud Run's 60-minute timeout constraint.

## Technical Context

**Language/Version**: Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`)
**Primary Dependencies**: google-adk[a2a]>=1.18.0, litellm>=1.65.0, httpx>=0.28.0, pydantic, google-cloud-secret-manager>=2.18.0
**Storage**: Filesystem JSON (evaluation results), Google Secret Manager (API keys)
**Testing**: pytest + pytest-asyncio (mock HTTP for integration tests)
**Target Platform**: Google Cloud Run (managed, serverless containers)
**Project Type**: Single project (extends existing `tau2_agent/`)
**Performance Goals**: Cold start < 30s, 60-minute request timeout (Cloud Run limit)
**Constraints**: num_tasks ≤ 30, num_trials ≤ 3 (per ADR-000), BYOK required for user simulator
**Scale/Scope**: 0-10 Cloud Run instances, $6-30/month server cost target

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: A2A/ADK/tau2 Compliance ✅

| Check | Status | Notes |
|-------|--------|-------|
| Extends tau2 via existing patterns | ✅ | Changes isolated to `tau2_agent/`, no core tau2 modifications (ADR-006) |
| Uses registry pattern | ✅ | tau2_agent already registered via ADK patterns |
| Message fidelity preserved | ✅ | A2A protocol unchanged, only add BYOK header extraction |
| Tool execution locality | ✅ | Tools still execute in tau2-bench process |

### Principle II: Backward Compatibility ✅

| Check | Status | Notes |
|-------|--------|-------|
| Zero breaking changes | ✅ | Local execution unchanged; GCP-specific code isolated |
| Existing tests pass | ✅ | No modifications to test suite or core behavior |
| CLI compatibility | ✅ | No CLI changes; Cloud Run is external deployment |
| BaseAgent interface unchanged | ✅ | A2AAgent implementation unchanged |

### Principle III: Metrics & Observability ⚠️

| Check | Status | Notes |
|-------|--------|-------|
| Token usage tracking | ⏸️ | Deferred to 007-datadog-project per spec |
| Execution time metrics | ⏸️ | Deferred to 007-datadog-project per spec |
| Protocol instrumentation | ⚠️ | Basic loguru logging for BYOK validation |

**Note**: Observability instrumentation explicitly deferred to 007-datadog-project. This spec focuses on deployment infrastructure only.

### Principle IV: Testing Philosophy ✅

| Check | Status | Notes |
|-------|--------|-------|
| Integration test coverage | ✅ | Add tests for middleware header extraction |
| Match tau2 test patterns | ✅ | Use existing pytest fixtures and patterns |
| No coverage gates | ✅ | Following tau2's pragmatic approach |

### Principle V: Code Quality Guidelines ✅

| Check | Status | Notes |
|-------|--------|-------|
| Type hints for public APIs | ✅ | New middleware/context modules will be typed |
| Async patterns | ✅ | FastAPI/ADK uses async; middleware is async-compatible |
| Error handling | ✅ | Return 400/401 errors with informative messages |
| Structured logging | ✅ | Use loguru; never log API keys |

### Principle VI: Architecture Principles ✅

| Check | Status | Notes |
|-------|--------|-------|
| Separation of concerns | ✅ | All GCP code in `tau2_agent/`, not core tau2 (ADR-006) |
| Registry pattern | ✅ | No registry changes needed |
| Configuration management | ✅ | Environment variables for server config, headers for BYOK |

### Principle VII: Documentation Standards ✅

| Check | Status | Notes |
|-------|--------|-------|
| Docstrings | ✅ | New modules will have Google-style docstrings |
| README examples | ✅ | Update tau2_agent/README.md with GCP deployment |

**Gate Result: PASS** - All principles satisfied or explicitly deferred with justification.

## Project Structure

### Documentation (this feature)

```text
specs/008-gcp-integration/
├── spec.md              # Feature specification
├── adr.md               # Architecture Decision Records (ADR-000 through ADR-007)
├── gcp-integration-guide.md  # GCP SDK/CLI/deployment reference
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── contracts/           # Phase 1 output (API contract)
│   └── byok-headers.yaml
└── quickstart.md        # Phase 1 output
```

### Source Code (repository root)

```text
tau2_agent/
├── agent.py                 # MODIFY: Use server-configured Gemini model
├── middleware.py            # CREATE: Extract X-User-LLM-* headers
├── context.py               # CREATE: Request context via contextvars
├── tools/
│   └── run_tau2_evaluation.py  # MODIFY: Read API key from context
├── docker_setup/
│   ├── Dockerfile           # MODIFY: Add GCP environment variables
│   ├── requirements.txt     # MODIFY: Add google-cloud-secret-manager
│   └── service.yaml         # CREATE: Cloud Run service definition
└── scripts/
    ├── deploy.sh            # CREATE: Cloud Run deployment script
    ├── setup-secrets.sh     # CREATE: Secret Manager setup script
    └── test-deployment.sh   # CREATE: Deployment verification script

tests/
├── test_tau2_agent/
│   ├── test_middleware.py   # CREATE: Middleware unit tests
│   └── test_context.py      # CREATE: Context tests
└── integration/
    └── test_byok_flow.py    # CREATE: BYOK integration test
```

**Structure Decision**: Single project extension of existing `tau2_agent/` directory. All GCP-specific code lives within `tau2_agent/` to maintain isolation from core `src/tau2/` per ADR-006.

## Complexity Tracking

> No violations requiring justification. Design follows existing patterns and ADR decisions.

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| No core tau2 changes | ADR-006 | tau2 already supports `llm_args_user={"api_key": ...}` |
| contextvars for BYOK | ADR-007 | Async-safe, request-scoped, no signature changes |
| Cloud Run limits | ADR-000 | num_tasks ≤ 30 ensures completion within 60-min timeout |
| BYOK via headers | ADR-004 | Separates auth from JSON-RPC payload, standard pattern |
