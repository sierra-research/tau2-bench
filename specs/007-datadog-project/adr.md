# ADR-001: Datadog Instrumentation Architecture for tau2-bench-agent

**Status**: Accepted
**Date**: 2025-12-23
**Decision Makers**: Architecture Review

## Context

The Datadog hackathon requires LLM observability for an application using Vertex AI or Gemini. We need to determine:

1. Whether tau2-bench-agent's architecture supports ddtrace instrumentation
2. What telemetry can be captured and where
3. What gaps exist and how to address them

## Decision Drivers

- **Hard Requirement**: LLM Observability signals must appear in Datadog
- **Hard Requirement**: Application health (latency/errors/tokens/cost) must be visible
- **Hard Requirement**: 3+ detection rules must trigger actionable items
- **Constraint**: Competition rules prohibit modifying core tau2-bench logic

## Architecture Analysis

### Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          tau2-bench Evaluation Flow                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  A2A Client ──► tau2_agent (ADK) ──► RunTau2Evaluation Tool                 │
│                                              │                               │
│                                              ▼                               │
│                                    ThreadPoolExecutor                        │
│                                              │                               │
│                                              ▼                               │
│                                    run_domain() [tau2-bench]                │
│                                              │                               │
│                    ┌─────────────────────────┼─────────────────────────┐    │
│                    │                         │                         │    │
│                    ▼                         ▼                         ▼    │
│            ┌───────────────┐         ┌───────────────┐         ┌───────────┐│
│            │ User Simulator│         │  A2AAgent     │         │Orchestrator│
│            │               │         │               │         │           ││
│            │ LiteLLM ──────┼────────►│ httpx ────────┼────────►│  (sync)   ││
│            │   ▼           │         │   ▼           │         │           ││
│            │ Gemini API    │         │ External Agent│         │           ││
│            └───────────────┘         └───────────────┘         └───────────┘│
│                   │                         │                               │
│                   │ ddtrace.patch()         │ ddtrace.patch()              │
│                   ▼                         ▼                               │
│            [LLM Spans]              [HTTP Spans]                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Instrumentation Points

| Component | Location | Framework | ddtrace Support |
|---|---|---|---|
| User Simulator → Gemini | `src/tau2/utils/llm_utils.py:209` | LiteLLM | ✅ `patch(litellm=True)` |
| A2A Client → External Agent | `src/tau2/a2a/client.py` | httpx | ✅ `patch(httpx=True)` |
| Evaluation orchestration | `src/tau2/run.py` | sync Python | ✅ Manual spans |
| Task execution | `src/tau2/orchestrator/orchestrator.py` | sync Python | ✅ Manual spans |

### Thread Context Propagation

**Finding**: `run_domain()` is called via `asyncio.run_in_executor()` in `run_tau2_evaluation.py:251-252`.

**Verified**: ddtrace 3.x+/4.x+ automatically propagates trace context to `ThreadPoolExecutor` threads. This means:
- LiteLLM calls inside the thread pool will be parented correctly
- HTTP spans for A2A calls will appear in the same trace
- No manual context propagation needed

## Considered Options

### Option 1: Auto-instrumentation Only (Chosen)

Add ddtrace with `patch(litellm=True, httpx=True)` and rely on automatic instrumentation.

**Pros:**
- Minimal code changes (~30 lines)
- Automatic LLM Observability for User Simulator
- Automatic HTTP tracing for A2A protocol
- Meets all hard requirements

**Cons:**
- No custom spans for tau2-specific operations (tasks, evaluations)
- Custom metrics require separate emission

### Option 2: Auto + Manual Instrumentation

Add ddtrace auto-instrumentation plus manual spans for tau2 operations.

**Pros:**
- Richer traces showing evaluation → task → LLM hierarchy
- Custom metrics embedded in spans

**Cons:**
- More code changes in tau2-bench core
- May conflict with competition "no modifications" rule

### Option 3: Post-hoc Metrics from Stored Data

Emit metrics from stored JSON files after evaluation completes.

**Pros:**
- Zero runtime overhead
- Works with existing data

**Cons:**
- No LLM Observability (fails hard requirement)
- No real-time visibility
- No trace correlation

## Decision

**Option 1: Auto-instrumentation Only** with supplementary custom metric emission.

**Updated 2025-12-24**: Changed from cli.py modification to wrapper-based approach to maintain zero-touch on tau2 core.

### Implementation

1. **Add ddtrace configuration module** (new file, ~130 lines):
   ```python
   # src/tau2/tracing.py
   def configure_ddtrace():
       from ddtrace import patch
       from ddtrace.llmobs import LLMObs

       patch(litellm=True, httpx=True)
       LLMObs.enable(ml_app="tau2-bench-agent", agentless_enabled=True)
   ```

2. **Wrapper-based activation** (zero tau2 core changes):
   ```python
   # src/experiments/datadog/scripts/tau2_traced.py
   # Configures ddtrace BEFORE importing tau2
   from tau2.tracing import configure_ddtrace
   configure_ddtrace()
   from tau2.cli import main as tau2_main
   tau2_main()
   ```

   **Usage options:**
   - `python -m experiments.datadog.scripts.tau2_traced run --domain mock` (recommended)
   - `ddtrace-run tau2 run --domain mock` (ddtrace's built-in wrapper)
   - `DD_PATCH_MODULES=litellm:true,httpx:true tau2 run --domain mock`

3. **Emit custom metrics** from evaluation results:
   ```python
   # After run_domain() completes
   from datadog import statsd
   statsd.gauge("tau2.task.reward", result["avg_reward"])
   ```

### Rationale for Wrapper Approach

The original plan to modify `cli.py` was rejected because:
- It adds `noqa` comments to suppress linting errors caused by early import
- It modifies a core tau2 module for an optional feature
- It violates the "minimal changes" principle

The wrapper approach:
- Keeps all Datadog code in `src/experiments/datadog/`
- Zero modifications to tau2 core modules
- Feature can be completely removed by deleting the experiment directory
- Follows standard ddtrace integration patterns

### What This Captures

| Signal | Source | Datadog Feature |
|---|---|---|
| Gemini API calls (prompt, completion, tokens, latency) | LiteLLM auto-instrumentation | LLM Observability |
| A2A protocol HTTP calls | httpx auto-instrumentation | APM Traces |
| Evaluation pass rate | Custom metric emission | Metrics Explorer |
| Task rewards | Custom metric emission | Metrics Explorer |
| Termination reasons | Custom metric emission | Metrics Explorer |

### What This Does NOT Capture

| Signal | Reason |
|---|---|
| External agent's internal LLM calls | Separate service, not instrumented |
| Per-message conversation traces | Would require A2A message-level spans |
| Real-time task-level progress | Evaluation runs synchronously |

## Consequences

### Positive
- Meets all hackathon hard requirements
- **Zero tau2 core modifications** - only adds `tracing.py` standalone module
- LLM Observability UI will show Gemini traces
- Detection rules can trigger on metrics
- Complies with "no core modifications" competition rule
- Entire feature can be disabled by not using the wrapper

### Negative
- No real-time per-task visibility during long evaluations
- External agent LLM calls not visible
- Custom metrics are post-hoc (after evaluation completes)

### Risks
- **ThreadPoolExecutor context loss**: Mitigated by ddtrace 1.x+ automatic propagation
- **LiteLLM integration gaps**: LiteLLM is officially supported by ddtrace
- **httpx async context**: ddtrace supports async httpx

## Validation

Before submission, verify:
1. `ddtrace-run tau2 run --domain mock` produces traces in Datadog APM
2. LLM Observability UI shows Gemini prompt/completion pairs
3. Custom metrics appear in Metrics Explorer
4. At least one monitor triggers and creates a Case

## Related Decisions

- **ADR-002**: Evaluation persistence for metrics emission (see below)
- **ADR-003** (pending): GCP deployment configuration (Cloud Run vs Cloud Functions)

---

# ADR-002: EvaluationStore Integration for Post-hoc Metrics Emission

**Status**: Accepted
**Date**: 2025-12-24
**Decision Makers**: Architecture Review

## Context

The `emit_metrics.py` script reads completed evaluation data from disk and emits metrics to Datadog. However, investigation revealed a gap in the current data flow:

### Current State (Problem)

| Evaluation Path | Output Location | emit_metrics.py Support |
|----------------|-----------------|------------------------|
| `tau2 run` CLI | `$TAU2_DATA_DIR/simulations/` | ❌ Script looks in `evaluations/` |
| `RunTau2Evaluation` tool | SSE events only (in-memory) | ❌ Nothing persisted to disk |
| `EvaluationStore` | `$TAU2_DATA_DIR/evaluations/` | ✅ Designed for this, but not integrated |

The `RunTau2Evaluation` tool (from 001-a2a-integration) was updated in 003-async-evaluation to use streaming utilities (`EvaluationProgress`, `create_adk_*_event`), but the **EvaluationStore integration for persistence was not implemented**.

Key finding in `tau2_agent/tools/run_tau2_evaluation.py:237`:
```python
config = RunConfig(
    ...
    save_to=None,  # Results NOT persisted
    ...
)
```

The tool generates an `evaluation_id` but never calls:
- `store.create_session()` to track the evaluation
- `store.complete_evaluation()` to persist results

### Implications for 007-datadog

Without persistence:
- Real-time ddtrace instrumentation works (LLM Observability, APM traces)
- Post-hoc custom metrics (`tau2.task.reward`, `tau2.evaluation.pass_rate`, etc.) cannot be emitted
- Detection rules based on custom metrics will not trigger

## Decision Drivers

- **Hard Requirement**: Custom metrics must be emittable for Datadog monitors/SLOs
- **Architectural Consistency**: Use existing EvaluationStore (002-evaluation-store) rather than creating parallel storage
- **Trace Correlation**: Evaluation ID should link real-time traces to post-hoc metrics

## Considered Options

### Option 1: Integrate EvaluationStore with RunTau2Evaluation (Chosen)

Add EvaluationStore calls to `RunTau2Evaluation._execute_streaming()`:
- Call `store.create_session()` at evaluation start
- Call `store.update_progress()` during evaluation (optional, for resume capability)
- Call `store.complete_evaluation()` with full results at end

**Pros:**
- Uses existing infrastructure (002-evaluation-store)
- Enables post-hoc metrics emission via `emit_metrics.py`
- Trace correlation via shared `evaluation_id`
- Session tracking for resume/recovery (future capability)

**Cons:**
- Requires changes to `RunTau2Evaluation` tool
- Additional dependency on store module

### Option 2: Update emit_metrics.py to read from simulations/

Have `emit_metrics.py` read from `$TAU2_DATA_DIR/simulations/` where `tau2 run` CLI saves.

**Pros:**
- Works immediately for CLI-based evaluations
- No changes to RunTau2Evaluation

**Cons:**
- Only works for CLI path, not A2A/ADK path
- Two different storage locations for same data type
- Doesn't enable metrics for primary use case (A2A evaluations)

### Option 3: Emit metrics in real-time during evaluation

Emit DogStatsD metrics from within `run_domain()` as tasks complete.

**Pros:**
- Real-time metrics visibility
- No post-hoc processing needed

**Cons:**
- Requires modifying tau2-bench core (violates ADR-001 principle)
- Tightly couples tau2-bench to Datadog
- Metrics emission failures could affect evaluation

## Decision

**Option 1: Integrate EvaluationStore with RunTau2Evaluation**

### Implementation Plan

1. **Update RunTau2Evaluation._execute_streaming()** in `tau2_agent/tools/run_tau2_evaluation.py`:

   ```python
   from tau2.store import create_store, EvaluationStatus

   async def _execute_streaming(self, ...):
       store = create_store()
       evaluation_id = generate_evaluation_id()

       # Create session at start
       store.create_session(
           evaluation_id=evaluation_id,
           domain=domain,
           agent_endpoint=agent_endpoint,
           # ... other metadata
       )

       try:
           result = await self._execute(...)

           # Complete with results
           store.complete_evaluation(
               evaluation_id=evaluation_id,
               results=EvaluationResults(
                   simulations=result["simulations"],  # Full simulation data
                   summary=result["summary"],
               )
           )
       except Exception as e:
           store.fail_evaluation(evaluation_id, error=str(e))
           raise
   ```

2. **Update emit_metrics.py** to read from `evaluations/` (already configured correctly)

3. **Ensure result includes full simulation data** for metrics extraction:
   - Currently `_execute()` returns summary only
   - Need to include `results.simulations` for per-task metrics

### Data Flow After Integration

```
RunTau2Evaluation
       │
       ├── generate_evaluation_id()
       │
       ├── store.create_session()  ──────► $TAU2_DATA_DIR/sessions/{id}.json
       │
       ├── run_domain() ──► ddtrace ──► Datadog APM/LLM Observability
       │
       ├── store.complete_evaluation() ──► $TAU2_DATA_DIR/evaluations/{id}.json
       │
       └── emit SSE events ──► A2A Client

                    Later:
                      │
emit_metrics.py ◄─────┘
       │
       └── DogStatsD ──► Datadog Metrics Explorer ──► Monitors/SLOs
```

## Consequences

### Positive
- Unified storage for all evaluation paths
- Enables post-hoc metrics for detection rules
- Trace correlation between real-time and post-hoc data
- Foundation for evaluation resume/recovery

### Negative
- Adds storage I/O to evaluation path
- RunTau2Evaluation becomes dependent on store module

### Risks
- **Storage failures**: Mitigate with try/catch - evaluation should succeed even if storage fails
- **Disk space**: Mitigated by existing retention policies in EvaluationStore

## Validation

After implementation, verify:
1. `RunTau2Evaluation` creates files in `$TAU2_DATA_DIR/evaluations/`
2. `emit_metrics.py --all` finds and processes these evaluations
3. Custom metrics appear in Datadog Metrics Explorer
4. Detection rules trigger based on metrics
