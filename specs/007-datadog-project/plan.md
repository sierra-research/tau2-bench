# Implementation Plan: Datadog LLM Observability Hackathon Project

**Branch**: `007-datadog-project` | **Date**: 2025-12-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/007-datadog-project/spec.md`
**Architecture**:
- [ADR-001](adr.md) - Datadog Instrumentation Architecture
- [ADR-002](adr.md#adr-002-evaluationstore-integration-for-post-hoc-metrics-emission) - EvaluationStore Integration for Post-hoc Metrics Emission
**Metrics Design**: [metrics_design.md](metrics_design.md) - Full metrics, detection rules, and remediation routing

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a Datadog-integrated LLM observability project for the Google Cloud Datadog hackathon. The project wraps tau2-bench-agent with ddtrace instrumentation to emit LLM Observability signals to Datadog, featuring 5+ detection rules that create actionable Cases/Incidents. Uses Gemini 2.0 Flash via LiteLLM, deployed to Cloud Run in agentless mode.

**Technical Approach** (per ADR-001):
1. Auto-instrumentation via `ddtrace.patch(litellm=True, httpx=True)`
2. LLM Observability via `LLMObs.enable()` in agentless mode (Cloud Run compatible)
3. Custom metrics emission post-evaluation via DogStatsD
4. Traffic generator script to demonstrate detection rules

## Technical Context

**Language/Version**: Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`)
**Primary Dependencies**:
- `ddtrace>=4.0.0` - Datadog tracing and LLM Observability (latest: 4.1.0)
- `datadog>=0.50.0` - Custom metrics and events via DogStatsD (latest: 0.52.1)
- `litellm` - LLM abstraction layer (auto-instrumented via `patch(litellm=True)`)
- `httpx` - Async HTTP client (auto-instrumented via `patch(httpx=True)`)
- `loguru` - Structured logging (matches tau2 convention)
**Storage**: Filesystem JSON (`$TAU2_DATA_DIR/evaluations/`) for post-hoc metrics emission
**Testing**: pytest + pytest-asyncio (matches tau2 convention)
**Target Platform**: Google Cloud Run (Linux container, 60-min max timeout)
**Project Type**: Single project with experiment directory pattern (`src/experiments/datadog/`)
**Performance Goals**: Complete evaluation of 30 tasks within 60-min Cloud Run timeout
**Constraints**:
- Max 30 tasks per evaluation (Cloud Run timeout limit)
- Agentless Datadog mode (no sidecar in serverless)
- Competition rule: tau2-bench-agent receives only ddtrace instrumentation, no feature changes
**Scale/Scope**: Demo scope - generate enough telemetry to trigger 5+ detection rules

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. A2A/ADK/tau2 Compliance ✅ PASS

| Requirement | Status | Notes |
|-------------|--------|-------|
| A2A Protocol Compliance | N/A | This feature adds observability, not A2A communication |
| ADK Integration | N/A | No ADK changes in this feature |
| tau2-bench Extension Pattern | ✅ | ddtrace integration via startup hook, no core modifications |
| Tool Execution Locality | N/A | No tool execution changes |

### II. Backward Compatibility ✅ PASS (NON-NEGOTIABLE)

| Requirement | Status | Notes |
|-------------|--------|-------|
| Zero Breaking Changes | ✅ | ddtrace is opt-in via `DD_TRACE_ENABLED=true` env var |
| Agent Registry | ✅ | No registry changes |
| CLI Compatibility | ✅ | No CLI flag changes - uses environment variables |
| BaseAgent Interface | ✅ | No interface changes |
| Data Model Compatibility | ✅ | No data model changes |

### III. Metrics & Observability ✅ IMPLEMENTS

| Requirement | Status | Notes |
|-------------|--------|-------|
| Token Usage Tracking | ✅ | LiteLLM auto-instrumentation captures tokens |
| Execution Time Metrics | ✅ | Custom `tau2.task.duration_seconds` metric |
| Protocol Instrumentation | ✅ | httpx auto-instrumentation for A2A calls |
| Metrics Export | ✅ | DogStatsD + Datadog Logs API |

### IV. Testing Philosophy ✅ PASS

| Requirement | Status | Notes |
|-------------|--------|-------|
| Pragmatic Integration Testing | ✅ | Follow tau2 approach - integration tests for traffic generator |
| Match tau2 Test Patterns | ✅ | Tests in `src/experiments/datadog/tests/` |
| No Coverage Gates | ✅ | No coverage requirements |

### V. Code Quality Guidelines ✅ PASS

| Requirement | Status | Notes |
|-------------|--------|-------|
| Type Hints for Public APIs | ✅ | Will use type hints in metrics emitter |
| Async Patterns | ✅ | Not required - metrics emission is post-hoc |
| Error Handling | ✅ | Graceful degradation if Datadog API fails |
| Structured Logging | ✅ | loguru with structured fields |

### VI. Architecture Principles ✅ PASS

| Requirement | Status | Notes |
|-------------|--------|-------|
| Separation of Concerns | ✅ | All Datadog code in `src/experiments/datadog/` |
| Interface Compliance | N/A | No new agent interfaces |
| Configuration Management | ✅ | Environment variables for all config |

### VII. Documentation Standards ✅ PASS

| Requirement | Status | Notes |
|-------------|--------|-------|
| Docstrings | ✅ | Google-style docstrings for metrics emitter |
| README Examples | ✅ | README with deployment instructions |
| Architecture Documentation | ✅ | ADR-001 already created |

### Gate Status: ✅ ALL GATES PASS

No violations requiring justification. The feature is additive and opt-in, with zero impact on existing tau2-bench functionality.

## Project Structure

### Documentation (this feature)

```text
specs/007-datadog-project/
├── spec.md              # Feature specification
├── adr.md               # ADR-001: Instrumentation architecture decision
├── metrics_design.md    # Full metrics, detection rules, case templates
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (metrics and logs schema)
├── quickstart.md        # Phase 1 output (local setup guide)
├── contracts/           # Phase 1 output (Datadog config JSON exports)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
# Component 1: tau2-bench-agent ddtrace integration (zero core modifications)
src/tau2/
└── tracing.py                    # NEW: ddtrace configuration module (~130 lines)

# Component 2: Hackathon experiment directory (extracted to new repo)
src/experiments/datadog/
├── README.md                     # Experiment overview + extraction instructions
├── configs/
│   ├── monitors.json             # Datadog monitor definitions (5+ detection rules)
│   ├── slos.json                 # SLO definitions
│   ├── dashboards.json           # Dashboard JSON exports
│   └── case_templates.json       # Case management templates
├── scripts/
│   ├── tau2_traced.py            # Wrapper script to run tau2 with tracing enabled
│   ├── traffic_generator.py      # Runs tau2 evaluations for telemetry
│   ├── emit_metrics.py           # Post-hoc metrics emission from JSON
│   ├── setup_datadog.py          # Creates monitors/dashboards via API
│   └── demo.sh                   # End-to-end demo script
├── deployment/
│   ├── Dockerfile                # Cloud Run deployment
│   ├── cloudbuild.yaml           # GCP Cloud Build config
│   └── requirements.txt          # Python dependencies
└── tests/
    ├── test_traffic_generator.py
    └── test_emit_metrics.py
```

**Structure Decision**: Two-component approach per spec.md:
1. **Zero tau2-bench core changes** - `tracing.py` is a standalone module, no cli.py modifications
2. **Wrapper-based activation** - Use `tau2_traced.py` or `ddtrace-run` to enable tracing
3. **Experiment directory** (`src/experiments/datadog/`) - extracted to standalone repo for submission

## Alignment with 008-gcp-integration

**Shared Components** (implemented once, used by both):

| Component | Location | Owner | Notes |
|-----------|----------|-------|-------|
| ddtrace configuration | `src/tau2/tracing.py` | **007** | 008's Cloud Run gets tracing if DD_TRACE_ENABLED=true |
| Cloud Run timeout limits | Config | **008** | 007 uses same limits (30 tasks max) |
| Environment variables | TAU2_AGENT_MODEL, GOOGLE_API_KEY | **008** | 007 uses same vars |

**007-Only Components** (not duplicated in 008):

| Component | Purpose | Why Separate |
|-----------|---------|--------------|
| `src/experiments/datadog/` | Hackathon wrapper | Competition rule: new repo required |
| monitors.json, slos.json | Datadog detection rules | Hackathon demo artifacts |
| traffic_generator.py | Generate telemetry | Demo for judges |
| emit_metrics.py | Post-hoc metrics | Hackathon-specific analytics |

**008-Only Components** (not duplicated in 007):

| Component | Purpose | Why Separate |
|-----------|---------|--------------|
| BYOK middleware | Client LLM credentials | 007 uses server-managed keys for demo |
| tau2_agent/middleware.py | Header extraction | Not needed for hackathon |
| tau2_agent/context.py | Request context | Not needed for hackathon |

**Implementation Order**:
1. **007 first**: Add `src/tau2/tracing.py` with ddtrace config
2. **008 inherits**: Cloud Run deployment automatically gets tracing capability

## EvaluationStore Integration (ADR-002)

**Critical Dependency**: Post-hoc metrics emission requires evaluation results to be persisted to disk. Per ADR-002, the `RunTau2Evaluation` tool must be updated to use EvaluationStore (from 002-evaluation-store).

### Current Gap

| Component | Status | Issue |
|-----------|--------|-------|
| `emit_metrics.py` | ✅ Reads from `evaluations/` | Expects data in `$TAU2_DATA_DIR/evaluations/` |
| `RunTau2Evaluation` | ❌ `save_to=None` | Results not persisted anywhere |
| `EvaluationStore` | ✅ Available | Not integrated with tool |

### Integration Requirements

1. **Update `_execute_streaming()`**:
   - Call `store.create_session()` at evaluation start
   - Include `evaluation_id` in session for trace correlation
   - Call `store.complete_evaluation()` with full results after completion
   - Call `store.fail_evaluation()` on error

2. **Update `_execute()`**:
   - Return `results.simulations` (full simulation data) in addition to summary
   - This enables per-task metrics extraction

3. **Data Flow After Integration**:
   ```
   RunTau2Evaluation
         │
         ├── store.create_session()  ──► $TAU2_DATA_DIR/sessions/{id}.json
         │
         ├── run_domain() ──► ddtrace ──► Datadog APM/LLM Observability
         │
         └── store.complete_evaluation() ──► $TAU2_DATA_DIR/evaluations/{id}.json
                                                      │
                                                      ▼
   emit_metrics.py ──► DogStatsD ──► Datadog Metrics Explorer
   ```

### Checkpoint Validation

Before proceeding to Phase 5:
1. Run evaluation via A2A: `curl -X POST .../run_tau2_evaluation`
2. Verify file exists: `ls $TAU2_DATA_DIR/evaluations/`
3. Run metrics emission: `python emit_metrics.py --all`
4. Verify metrics in Datadog Metrics Explorer

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

✅ No violations - all gates pass. The design follows the principle of minimal changes to tau2-bench-agent core.
