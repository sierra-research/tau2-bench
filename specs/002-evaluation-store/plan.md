# Implementation Plan: Evaluation Store

**Branch**: `002-evaluation-store` | **Date**: 2025-12-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-evaluation-store/spec.md`

## Summary

Implement filesystem-based evaluation storage with atomic writes, flat directory for completed evaluations, mutable session tracking for in-progress evaluations, and structured logging. Separate from existing `simulations/` to avoid modifying CLI behavior. Supports OTel/Datadog correlation via trace_id and session_id fields for integration with 003-async-evaluation and 006-otel-integration.

## Technical Context

**Language/Version**: Python 3.10+ (matches tau2-bench pyproject.toml requires-python)
**Primary Dependencies**: pydantic (data models), loguru (logging), pathlib (file operations)
**Storage**: Filesystem JSON files in `$TAU2_DATA_DIR` (default `./data`)
**Testing**: pytest + pytest-asyncio (matches tau2 conventions)
**Target Platform**: Linux server (Docker containers, CI)
**Project Type**: Single project - extends existing `src/tau2/` structure
**Performance Goals**: List 5000 evaluations in <500ms, atomic writes for data integrity
**Constraints**: No external databases, single node storage, immutable completed records
**Scale/Scope**: Up to 5000 users, 30-day retention for completed evaluations

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. A2A/ADK/tau2 Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| Follows tau2 extension pattern | ✅ PASS | New module `src/tau2/store/` alongside existing modules |
| Preserves message fidelity | ✅ N/A | Store persists evaluations, doesn't translate messages |
| Tools execute in tau2 process | ✅ N/A | No tool execution in store module |

### II. Backward Compatibility

| Requirement | Status | Notes |
|-------------|--------|-------|
| Zero breaking changes | ✅ PASS | New module, no changes to existing code |
| All existing tests pass | ✅ PASS | No modifications to existing tests |
| Evaluation results reproducible | ✅ PASS | Read-only integration, doesn't affect evaluation logic |
| CLI compatibility | ✅ PASS | No CLI changes in this feature |
| BaseAgent interface unchanged | ✅ N/A | Store doesn't implement BaseAgent |
| Data model compatibility | ✅ PASS | Uses existing Message types for serialization |

### III. Metrics & Observability

| Requirement | Status | Notes |
|-------------|--------|-------|
| Token tracking | ✅ PASS | Stores token metrics in evaluation results |
| Execution time metrics | ✅ PASS | Stores timing in state_history and completed_at |
| Protocol instrumentation | ✅ PASS | Structured logging with loguru |
| Metrics export | ✅ PASS | JSON files are directly exportable |

### IV. Testing Philosophy

| Requirement | Status | Notes |
|-------------|--------|-------|
| Pragmatic integration tests | ✅ PASS | Will add 5-10 integration tests |
| Match tau2 test patterns | ✅ PASS | Tests in `tests/test_store/` using pytest |
| No coverage gates | ✅ PASS | No coveragerc or percentage requirements |

### V. Code Quality Guidelines

| Requirement | Status | Notes |
|-------------|--------|-------|
| Type hints for public APIs | ✅ PASS | All public functions typed |
| Error handling with context | ✅ PASS | Domain-specific exceptions (EvaluationIdCollisionError) |
| Structured logging | ✅ PASS | loguru with evaluation_id, trace_id fields |

### VI. Architecture Principles

| Requirement | Status | Notes |
|-------------|--------|-------|
| Separation of concerns | ✅ PASS | New `src/tau2/store/` module |
| No core tau2 imports | ✅ PASS | Core code doesn't import store module |
| Configuration management | ✅ PASS | Environment variables for all config |

### VII. Documentation Standards

| Requirement | Status | Notes |
|-------------|--------|-------|
| Google-style docstrings | ✅ PASS | All public APIs documented |
| Type hints in docstrings | ✅ PASS | Complex types explained |

**Gate Status**: ✅ PASS - All constitution requirements satisfied. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/002-evaluation-store/
├── plan.md              # This file
├── research.md          # Phase 0 output - atomic writes, retention patterns
├── data-model.md        # Phase 1 output - Evaluation, Session, Event schemas
├── quickstart.md        # Phase 1 output - usage examples
├── contracts/           # Phase 1 output - Python interface contracts
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Data Directory (runtime)

```text
data/
├── simulations/              # EXISTING - CLI batch runs (untouched)
├── evaluations/              # NEW - Completed API evaluations (flat)
├── sessions/                 # NEW - In-progress tracking (temporary)
└── logs/                     # NEW - Structured event logs
```

### Source Code (repository root)

```text
src/tau2/
├── store/                     # NEW: Evaluation store module
│   ├── __init__.py           # Public API exports
│   ├── store.py              # Core storage operations (~100 lines)
│   ├── retention.py          # Cleanup and retention logic (~60 lines)
│   ├── events.py             # Structured event logging (~40 lines)
│   ├── models.py             # Pydantic models for evaluation data
│   └── exceptions.py         # Custom exceptions
├── a2a/                      # Existing A2A module
├── agent/                    # Existing agents
├── api_service/              # Existing API service
└── ...                       # Other existing modules

tests/
├── test_store/               # NEW: Store module tests
│   ├── test_store.py         # Storage operations tests
│   ├── test_retention.py     # Retention and cleanup tests
│   └── test_events.py        # Event logging tests
└── ...                       # Other existing tests
```

**Structure Decision**: Single project structure following existing `src/tau2/` layout. New `store/` module added alongside existing modules (a2a, agent, api_service). Test structure mirrors source in `tests/test_store/`.

## Complexity Tracking

> No violations requiring justification. Design uses simple filesystem operations with no external databases or complex abstractions.

---

## Post-Design Constitution Re-Check

*Re-evaluated after Phase 1 design artifacts completed.*

### Design Validation

| Principle | Pre-Design | Post-Design | Change |
|-----------|------------|-------------|--------|
| I. A2A/ADK/tau2 Compliance | ✅ PASS | ✅ PASS | No change |
| II. Backward Compatibility | ✅ PASS | ✅ PASS | No change |
| III. Metrics & Observability | ✅ PASS | ✅ PASS | No change |
| IV. Testing Philosophy | ✅ PASS | ✅ PASS | No change |
| V. Code Quality Guidelines | ✅ PASS | ✅ PASS | No change |
| VI. Architecture Principles | ✅ PASS | ✅ PASS | No change |
| VII. Documentation Standards | ✅ PASS | ✅ PASS | No change |

### Design Artifacts Validation

| Artifact | Status | Notes |
|----------|--------|-------|
| research.md | ✅ Complete | 6 decisions documented with rationale |
| data-model.md | ✅ Complete | 8 entities, validation rules, state machine |
| contracts/ | ✅ Complete | 3 protocols, 8 models, 4 exceptions |
| quickstart.md | ✅ Complete | Usage examples, integration patterns |

### Final Gate Status

✅ **PASS** - All constitution checks pass. Design is complete and ready for task generation.

**Next Step**: Run `/speckit.tasks` to generate implementation tasks.
