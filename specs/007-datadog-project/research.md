# Research: Datadog LLM Observability Hackathon Project

**Feature Branch**: `007-datadog-project`
**Date**: 2025-12-24

## Research Tasks Completed

### 1. ddtrace LiteLLM Integration

**Decision**: Use `ddtrace.patch(litellm=True)` for automatic LLM Observability

**Rationale**:
- LiteLLM is officially supported by ddtrace (see [dd-trace-py docs](https://ddtrace.readthedocs.io/en/stable/integrations.html#llm-observability))
- Captures prompts, completions, token counts, and latency automatically
- No code changes required in tau2-bench LLM calls

**Alternatives Considered**:
- Manual span creation: Rejected - requires modifying core tau2 code
- OpenTelemetry export: Rejected - Datadog LLM Observability requires native integration

**Verification Points** (from ADR-001):
- Thread pool context propagation: ddtrace 1.x+ automatically propagates trace context to `ThreadPoolExecutor` threads
- LiteLLM calls inside `run_domain()` will be correctly parented in the trace

### 2. Datadog Deployment Mode

**Decision**: Agentless mode via `LLMObs.enable(agentless_enabled=True)`

**Rationale**:
- Cloud Run is a serverless environment - no sidecar containers allowed
- Agentless mode sends data directly to Datadog intake API
- Requires `DD_API_KEY` and `DD_SITE` environment variables

**Alternatives Considered**:
- Datadog Agent sidecar: Rejected - not supported in Cloud Run
- DaemonSet: Rejected - Cloud Run doesn't support DaemonSets

**Configuration**:
```python
LLMObs.enable(
    ml_app=os.getenv("DD_SERVICE", "tau2-bench-agent"),
    agentless_enabled=True,
)
```

### 3. Custom Metrics Strategy

**Decision**: Post-hoc emission from stored evaluation JSON using DogStatsD

**Rationale** (from metrics_design.md):
- Evaluation runs synchronously - metrics emitted after completion
- DogStatsD provides reliable metric submission
- JSON files contain all reward_info, assertions, and timing data

**Alternatives Considered**:
- Real-time streaming: Rejected - evaluation is synchronous, no streaming support
- Span tags only: Rejected - custom metrics needed for monitors and SLOs

**Implementation**: `emit_metrics.py` reads `$TAU2_DATA_DIR/evaluations/*.json` and emits:
- Task-level: `tau2.task.reward`, `tau2.task.duration_seconds`, `tau2.task.success`
- Tool-level: `tau2.tool.calls`, `tau2.tool.correct`
- Assertion-level: `tau2.assertion.result`
- Termination: `tau2.termination` with reason tag

### 4. Detection Rules Design

**Decision**: 5 detection rules with DR-002 (Task Failure Spike) as "hero" monitor

**Rationale** (from spec.md clarifications):
- DR-002 most relevant to LLM quality (avg reward <0.7 in 10 min)
- Traffic generator prioritizes triggering low-reward scenarios
- Full Case creation workflow demonstrated for judges

**Rules Defined** (per metrics_design.md):
| ID | Name | Query Summary | Action |
|---|---|---|---|
| DR-001 | High Error Rate | error_count / total > 0.2 | Create Case |
| DR-002 | Task Failure Spike ⭐ | avg:tau2.task.reward < 0.7 | Create Case |
| DR-003 | Token Cost Anomaly | token_cost > 2x baseline | Alert |
| DR-004 | Premature Termination | termination:max_errors > 10/hr | Create Incident |
| DR-005 | Latency SLO Breach | p99:duration > 60s | SLO Alert |

### 5. Cloud Run Deployment Limits

**Decision**: Max 30 tasks per evaluation, gemini-2.0-flash model

**Rationale** (from spec.md alignment with 008-gcp-integration):
- Cloud Run 60-minute maximum request timeout
- Average task times: ~40s (mock), ~64s (airline), ~69s (retail)
- 30 tasks × ~60s = ~30 min, safe margin for timeout

**Constraints**:
| Parameter | Limit | Rationale |
|---|---|---|
| num_tasks | Max 30 | ~30-40 min execution, safe margin |
| num_trials | Max 3 | Multiplies execution time |

### 6. Traffic Generator Failure Scenarios

**Decision**: Include scripted failure scenarios to trigger detection rules

**Rationale** (from spec.md clarifications):
- Judges need to see detection rules in action
- Traffic generator includes intentional failure modes:
  - Invalid tool call patterns (triggers MAX_ERRORS termination)
  - High-latency scenarios (triggers Latency SLO breach)
  - Forced low-reward tasks (triggers Task Failure Spike monitor)

**Implementation**: Traffic generator script runs:
1. Normal evaluations (baseline metrics)
2. Failure mode evaluations (trigger monitors)
3. Recovery evaluations (demonstrate Case resolution)

### 7. Repository Structure

**Decision**: `src/experiments/datadog/` for development, extract to standalone repo for submission

**Rationale** (from spec.md):
- Competition rule: "Project must be Your original creation"
- tau2-bench-agent receives only ddtrace instrumentation
- Hackathon submission is a separate repository wrapping tau2

**Extraction Path**:
```bash
git subtree split -P src/experiments/datadog -b datadog-standalone
git push git@github.com:wuTims/tau2-datadog-observability.git datadog-standalone:main
```

## Dependencies Verified

| Dependency | Version | Purpose | Verified |
|---|---|---|---|
| ddtrace | >=4.0.0 | Tracing and LLM Observability | ✅ [PyPI](https://pypi.org/project/ddtrace/) (latest: 4.1.0, 2025-12-19) |
| datadog | >=0.50.0 | DogStatsD metrics and events | ✅ [PyPI](https://pypi.org/project/datadog/) (latest: 0.52.1, 2025-07-31) |
| litellm | existing | LLM abstraction (already in tau2) | ✅ Auto-instrumented via `patch(litellm=True)` |
| httpx | existing | Async HTTP (already in tau2) | ✅ Auto-instrumented via `patch(httpx=True)` |

**Note**: ddtrace 4.x dropped Python 3.8 support (requires 3.9+). This aligns with our Python 3.10+ requirement.
The `google_generativeai` integration was removed in ddtrace 4.0; use `google_genai` instead if direct Gemini SDK usage is needed (we use LiteLLM which is unaffected).

## Open Items Resolved

| Question | Resolution | Source |
|---|---|---|
| Which Datadog deployment mode? | Agentless | spec.md Session 2025-12-24 |
| Include failure scenarios? | Yes, scripted | spec.md Session 2025-12-24 |
| Handle external dependency failures? | Log and continue | spec.md Session 2025-12-24 |
| Automated failure classification? | No, manual only | spec.md Session 2025-12-24 |
| Hero detection rule? | DR-002 Task Failure Spike | spec.md Session 2025-12-24 |

## Research Complete

All NEEDS CLARIFICATION items from Technical Context have been resolved. Proceed to Phase 1: Design & Contracts.
