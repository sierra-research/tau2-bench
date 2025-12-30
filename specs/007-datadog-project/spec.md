# Feature Specification: Datadog LLM Observability Hackathon Project

**Feature Branch**: `007-datadog-project`
**Created**: 2025-12-21
**Updated**: 2025-12-23
**Status**: Draft (Architecture Verified)
**Depends On**: None (independent project)
**Aligned With**: 008-gcp-integration (tau2_agent service setup, Gemini integration)
**Architecture Decisions**:
- [ADR-001](adr.md) - Datadog Instrumentation Architecture
- [ADR-002](adr.md#adr-002-evaluationstore-integration-for-post-hoc-metrics-emission) - EvaluationStore Integration for Post-hoc Metrics Emission
**Metrics Design**: [metrics_design.md](metrics_design.md) - Full metrics, detection rules, and remediation routing

**Input**: "Create a Datadog-integrated LLM observability project for the Google Cloud Datadog hackathon challenge, using tau2-bench-agent as the application with Gemini models"

## Problem Statement

The Datadog hackathon challenge requires demonstrating end-to-end LLM observability with:
1. An LLM application powered by Vertex AI or Gemini
2. Telemetry streamed to Datadog (traces, metrics, logs)
3. Detection rules that trigger actionable items
4. Dashboards showing application health

tau2-bench-agent is an ideal candidate because:
- It orchestrates complex multi-turn agent conversations
- It evaluates agent performance with rich metrics (pass^k, rewards)
- It generates natural "traffic" via benchmark evaluations
- It has clear failure modes that map to detection rules

## Competition Constraints

**Critical Rule**: "The Project must be Your original creation not a modification or extension of Your or anyone else's existing work."

This means:
- tau2-bench-agent can ONLY receive ddtrace instrumentation (no new features)
- The hackathon submission must be a **separate repository/project**
- The new project hosts/wraps tau2-bench with Datadog integration

## Architecture Overview

### LLM Roles in tau2-bench Evaluation

Per the GCP Integration spec (008), tau2-bench evaluations involve three LLM roles:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        tau2-bench Evaluation Flow                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐ │
│  │  tau2_agent LLM  │     │ User Simulator   │     │  Agent Under     │ │
│  │  (Orchestrator)  │     │     LLM          │     │   Evaluation     │ │
│  ├──────────────────┤     ├──────────────────┤     ├──────────────────┤ │
│  │ Purpose:         │     │ Purpose:         │     │ Purpose:         │ │
│  │ - Orchestrate    │     │ - Simulate user  │     │ - Handle tasks   │ │
│  │   evaluation     │     │   interactions   │     │ - Being tested   │ │
│  │ - Analyze traces │     │ - Generate       │     │                  │ │
│  │ - Report results │     │   realistic      │     │                  │ │
│  │                  │     │   requests       │     │                  │ │
│  ├──────────────────┤     ├──────────────────┤     ├──────────────────┤ │
│  │ Cost: Server     │     │ Cost: SERVER     │     │ Cost: External   │ │
│  │ (default Gemini) │     │ (Hackathon demo) │     │ (not our concern)│ │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Note**: For the hackathon demo, we pay for both the tau2_agent LLM and User Simulator LLM using server-managed Gemini. This differs from the production GCP deployment (008-gcp-integration) which uses BYOK for the User Simulator.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HACKATHON SUBMISSION REPO                         │
│                    (tau2-datadog-observability)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Google Cloud Platform                        │ │
│  ├────────────────────────────────────────────────────────────────┤ │
│  │  ┌─────────────────────┐     ┌───────────────────────────────┐ │ │
│  │  │   Cloud Run         │     │      Gemini Developer API     │ │ │
│  │  │                     │     │                               │ │ │
│  │  │  tau2_agent         │────►│  Orchestrator: gemini-2.0-flash│ │ │
│  │  │  + ddtrace layer    │     │  User Sim:     gemini-2.0-flash│ │ │
│  │  │                     │     │                               │ │ │
│  │  └──────────┬──────────┘     └───────────────────────────────┘ │ │
│  │             │                                                   │ │
│  │             │ ddtrace auto-instrumentation                      │ │
│  └─────────────┼───────────────────────────────────────────────────┘ │
│                │                                                      │
│                ▼ Telemetry (traces, metrics, logs)                   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Datadog Platform                             │ │
│  ├────────────────────────────────────────────────────────────────┤ │
│  │                                                                 │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐   │ │
│  │  │ LLM Observ.  │ │   APM        │ │   Custom Metrics     │   │ │
│  │  │              │ │              │ │                      │   │ │
│  │  │ - Gemini     │ │ - Traces     │ │ - tau2.pass_rate     │   │ │
│  │  │   traces     │ │ - Errors     │ │ - tau2.avg_reward    │   │ │
│  │  │ - Token      │ │ - Latency    │ │ - tau2.error_rate    │   │ │
│  │  │   usage      │ │              │ │ - tau2.task_duration │   │ │
│  │  │ - Cost       │ │              │ │                      │   │ │
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘   │ │
│  │         │                │                    │                │ │
│  │         └────────────────┼────────────────────┘                │ │
│  │                          ▼                                      │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │                   Detection Rules                         │ │ │
│  │  ├──────────────────────────────────────────────────────────┤ │ │
│  │  │ 1. High Error Rate Monitor (>20% in 5 min)               │ │ │
│  │  │ 2. Task Failure Spike Monitor (reward=0 >30% in 10 min)  │ │ │
│  │  │ 3. Token Cost Anomaly Monitor (>2x baseline)             │ │ │
│  │  │ 4. Latency SLO (p99 < 60s)                               │ │ │
│  │  │ 5. Premature Termination Alert (MAX_ERRORS spike)        │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  │         │                                                      │ │
│  │         ▼                                                      │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │              Case / Incident Management                   │ │ │
│  │  │  - Auto-create cases for warnings                        │ │ │
│  │  │  - Auto-create incidents for critical failures           │ │ │
│  │  │  - Include task_id, domain, error context                │ │ │
│  │  │  - Link to runbook                                       │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │            Dashboard: tau2-bench Health                   │ │ │
│  │  │  - Pass rate by domain                                   │ │ │
│  │  │  - Avg reward trend                                      │ │ │
│  │  │  - LLM latency/token metrics                             │ │ │
│  │  │  - Error breakdown by termination reason                 │ │ │
│  │  │  - Active cases/incidents                                │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Verified Instrumentation Architecture

> **See Also**: [ADR-001: Datadog Instrumentation Architecture](adr.md) for detailed decision rationale.

### Architecture Verification (2025-12-23)

Analysis of the tau2-bench-agent codebase confirms ddtrace instrumentation is feasible:

#### Instrumentation Points

| Component | Code Location | Framework | ddtrace Method |
|---|---|---|---|
| **User Simulator → Gemini** | `src/tau2/utils/llm_utils.py:209` | LiteLLM `completion()` | `patch(litellm=True)` ✅ |
| **A2A Client → External Agent** | `src/tau2/a2a/client.py` | httpx `AsyncClient` | `patch(httpx=True)` ✅ |
| **Evaluation Orchestration** | `src/tau2/run.py` | sync Python | Manual spans (optional) |
| **RunTau2Evaluation Tool** | `tau2_agent/tools/run_tau2_evaluation.py` | ADK Tool | Automatic span parenting |

#### Thread Pool Context Propagation

**Critical Finding**: `run_domain()` is called via `ThreadPoolExecutor` (`run_tau2_evaluation.py:251-252`):

```python
loop = asyncio.get_running_loop()
results = await loop.run_in_executor(None, run_domain, config)
```

**Verification**: ddtrace 1.x+ automatically propagates trace context to `ThreadPoolExecutor` threads. LiteLLM calls inside the thread pool will be correctly parented in the trace.

#### What Gets Captured

| Signal | Source | Datadog Feature |
|---|---|---|
| Gemini prompts/completions | LiteLLM auto-instrumentation | LLM Observability UI |
| Token counts & costs | LiteLLM auto-instrumentation | LLM Observability Metrics |
| A2A protocol HTTP calls | httpx auto-instrumentation | APM Traces |
| Evaluation latency | Span timing | APM Latency |
| Custom tau2 metrics | Post-evaluation emission | Metrics Explorer |

#### What Does NOT Get Captured

| Signal | Reason | Mitigation |
|---|---|---|
| External agent's internal LLM calls | Separate service | Agent owner must instrument |
| Real-time per-task progress | Evaluation is synchronous | Use SSE events (003-async-evaluation) |
| Per-message A2A conversation traces | Not implemented | Future enhancement |

#### LLMObs Evaluation Integration

The `tau2_agent/llmobs_evaluations.py` module submits evaluation metrics directly to Datadog LLM Observability using `LLMObs.submit_evaluation()`. This provides trace-correlated quality metrics that appear alongside LLM calls in the Datadog UI.

**Opt-in Behavior**: Evaluations are only submitted when both environment variables are set:
- `DD_TRACE_ENABLED=true`
- `DD_LLMOBS_ENABLED=true`

**Metrics Submitted**:

| Label | Type | Value | Description |
|-------|------|-------|-------------|
| `tau2.task.reward` | score | 0.0-1.0 | Per-task reward score |
| `tau2.task.success` | categorical | pass/fail | Binary success (reward >= 0.7) |
| `tau2.task.termination` | categorical | reason | Why task ended |
| `tau2.assertion.db_check` | categorical | pass/fail | Database state validation |
| `tau2.assertion.nl_pass_rate` | score | 0.0-1.0 | NL assertion pass ratio |
| `tau2.assertion.action_accuracy` | score | 0.0-1.0 | Tool call correctness ratio |
| `tau2.assertion.communicate_pass_rate` | score | 0.0-1.0 | Communication check pass ratio |
| `tau2.evaluation.pass_rate` | score | 0.0-1.0 | Overall evaluation pass rate |
| `tau2.evaluation.avg_reward` | score | 0.0-1.0 | Average reward across tasks |

**Complementary Telemetry Channels**:

| Channel | Purpose | UI Location |
|---------|---------|-------------|
| `llmobs_evaluations.py` | Real-time trace-correlated evaluations | LLM Observability → Traces |
| `emit_metrics.py` | Post-hoc aggregated metrics for monitors/SLOs | Metrics Explorer, Dashboards |

The two channels serve different purposes: LLMObs evaluations enable drill-down from traces to quality metrics, while DogStatsD metrics power detection rules and dashboards.

#### EvaluationStore Integration Dependency (ADR-002)

**Critical Finding**: The `RunTau2Evaluation` tool currently generates SSE events but does NOT persist evaluation results to disk. Per ADR-002:

| Evaluation Path | Output Location | Post-hoc Metrics Support |
|----------------|-----------------|------------------------|
| `tau2 run` CLI | `$TAU2_DATA_DIR/simulations/` | ❌ Script looks in `evaluations/` |
| `RunTau2Evaluation` tool | SSE events only (in-memory) | ❌ Nothing persisted |
| **With EvaluationStore** | `$TAU2_DATA_DIR/evaluations/` | ✅ Full support |

**Implication**: Phase 4.5 (EvaluationStore Integration) MUST be completed before Phase 5 (Datadog Configuration) to enable post-hoc metrics emission via `emit_metrics.py`.

#### Hackathon Requirement Compliance

| Hard Requirement | Status | How |
|---|---|---|
| LLM Observability signals | ✅ Met | LiteLLM auto-instrumentation captures Gemini calls |
| Application health (latency/errors/tokens/cost) | ✅ Met | Combination of APM spans + custom metrics |
| 3+ detection rules | ✅ Met | Monitors on custom metrics |
| Actionable records (Case/Incident) | ✅ Met | Monitor → Case Management workflow |
| Vertex AI / Gemini | ✅ Met | LiteLLM with `gemini/` prefix |

## Local Development Structure

### Decision: `src/experiments/datadog/`

All datadog hackathon code lives in `src/experiments/datadog/` within this repository during development.

**Rationale**:
- The `experiments/` directory is designed for self-contained experimental code
- Enables development with full tau2-bench-agent context
- Clean extraction path via `git subtree split` when ready to publish
- Follows existing patterns (`agentify_tau_bench`, `hyperparam`)

**Directory Structure**:

```
src/experiments/datadog/
├── README.md                    # Experiment overview + extraction instructions
├── configs/
│   ├── monitors.json            # Datadog monitor definitions
│   ├── slos.json                # SLO definitions
│   ├── dashboards.json          # Dashboard JSON exports
│   └── case_templates.json      # Case management templates
├── scripts/
│   ├── traffic_generator.py     # Runs tau2 evaluations for telemetry
│   ├── setup_datadog.py         # Creates monitors/dashboards via API
│   └── demo.sh                  # End-to-end demo script
├── deployment/
│   ├── Dockerfile               # Cloud Run deployment
│   ├── cloudbuild.yaml          # GCP Cloud Build config
│   └── requirements.txt         # Python dependencies
└── tests/
    └── test_traffic_generator.py
```

**Extraction to Standalone Repo**:

When ready for hackathon submission:

```bash
# Extract datadog directory with history
git subtree split -P src/experiments/datadog -b datadog-standalone

# Push to new repository
git push git@github.com:wuTims/tau2-datadog-observability.git datadog-standalone:main
```

**What Stays vs What Moves**:

| Component | Location | Moves to New Repo? |
|-----------|----------|-------------------|
| ddtrace configuration | `src/tau2/tracing.py` | No (tau2 integration) |
| Datadog configs | `src/experiments/datadog/configs/` | Yes |
| Traffic generator | `src/experiments/datadog/scripts/` | Yes |
| Deployment files | `src/experiments/datadog/deployment/` | Yes |
| Tests | `src/experiments/datadog/tests/` | Yes |

## Two-Repository Strategy

### Repository 1: tau2-bench-agent (Existing - Minimal Changes)

**Branch**: `feature/ddtrace-instrumentation`

Only changes:
1. Add `ddtrace` to dependencies
2. Add ddtrace startup configuration
3. Add span annotations to key methods (non-invasive)

```python
# pyproject.toml - ONLY ADDITION
dependencies = [
    # ... existing deps ...
    "ddtrace>=4.0.0",  # Latest: 4.1.0 (requires Python 3.9+)
]
```

```python
# src/tau2/tracing.py - NEW FILE (minimal, ~30 lines)
"""Datadog tracing configuration for tau2-bench-agent."""

import os

def configure_ddtrace():
    """Configure ddtrace if enabled via environment."""
    if not os.getenv("DD_TRACE_ENABLED", "false").lower() == "true":
        return

    from ddtrace import tracer, patch
    from ddtrace.llmobs import LLMObs

    # Enable LLM Observability for Gemini (via LiteLLM)
    # Using agentless mode for Cloud Run compatibility (no sidecar needed)
    if os.getenv("DD_LLMOBS_ENABLED", "true").lower() == "true":
        LLMObs.enable(
            ml_app=os.getenv("DD_SERVICE", "tau2-bench-agent"),
            agentless_enabled=True,  # Required for Cloud Run - sends directly to Datadog intake
        )

    # Patch frameworks for auto-instrumentation:
    # - litellm: Captures User Simulator → Gemini LLM calls
    # - httpx: Captures A2A Client → External Agent HTTP calls
    patch(litellm=True, httpx=True)

    # Set service-level tags for filtering in Datadog
    tracer.set_tags({
        "tau2.version": os.getenv("TAU2_VERSION", "unknown"),
    })
```

```python
# src/experiments/datadog/scripts/tau2_traced.py - WRAPPER SCRIPT (no cli.py changes)
# Configures ddtrace BEFORE importing tau2
from tau2.tracing import configure_ddtrace
configure_ddtrace()

from tau2.cli import main as tau2_main
tau2_main()
```

**Note**: The wrapper-based approach was chosen to maintain zero modifications to tau2 core modules. See ADR-001 for rationale.

### Repository 2: tau2-datadog-observability (New - Hackathon Submission)

**Purpose**: Wraps tau2-bench-agent with full Datadog integration for the hackathon.

```
tau2-datadog-observability/
├── README.md                    # Deployment instructions
├── LICENSE                      # Apache-2.0 or MIT
├── Dockerfile                   # Cloud Run deployment
├── cloudbuild.yaml              # GCP Cloud Build
├── requirements.txt             # Dependencies
├── datadog/
│   ├── monitors.json            # Exported monitor definitions
│   ├── slos.json                # Exported SLO definitions
│   ├── dashboards.json          # Exported dashboard JSON
│   └── case_templates.json      # Case management templates
├── scripts/
│   ├── traffic_generator.py     # Runs tau2 evaluations
│   ├── setup_datadog.py         # Creates monitors/dashboards via API
│   └── demo.sh                  # End-to-end demo script
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py              # FastAPI wrapper (optional health endpoint)
│       └── config.py            # Environment configuration
└── tests/
    └── test_traffic_generator.py
```

## Functional Requirements

### FR-001: Gemini Integration (Aligned with 008-gcp-integration)

Per the GCP Integration spec, the service uses **Gemini Developer API** (not Vertex AI):

| Component | Model | Cost Bearer | Notes |
|-----------|-------|-------------|-------|
| **tau2_agent** (Orchestrator) | `gemini-2.0-flash` | Server | Low cost, server-managed |
| **User Simulator** | `gemini-2.0-flash` | Server (for demo) | Hackathon absorbs cost |
| **Agent Under Evaluation** | External | External | Not managed by us |

**Configuration**:
- Both LLMs use server-managed `GOOGLE_API_KEY` environment variable
- Model configured via `TAU2_AGENT_MODEL=gemini-2.0-flash`
- LiteLLM used for model abstraction with `gemini/` prefix

### FR-002: Datadog Telemetry Emission
- ddtrace SHALL auto-instrument LiteLLM calls to Gemini
- LLM Observability spans SHALL capture: model, tokens, latency, cost
- Custom metrics SHALL be emitted for tau2-specific data:
  - `tau2.task.reward` (gauge) - Task reward value
  - `tau2.task.duration` (histogram) - Task execution time
  - `tau2.task.error_count` (count) - Errors per task
  - `tau2.evaluation.pass_rate` (gauge) - Pass^k rate
  - `tau2.termination.reason` (count, tagged) - Termination breakdown

### FR-003: Detection Rules
The project SHALL define at least 5 detection rules:

| Rule ID | Name | Query | Threshold | Action |
|---------|------|-------|-----------|--------|
| DR-001 | High Error Rate | `sum:tau2.task.success{success:false} / sum:tau2.task.total` | >20% in 5m | Create Case |
| **DR-002** | **Task Failure Spike** ⭐ | `avg:tau2.task.reward{*}` | <0.7 in 10m | Create Case |
| DR-003 | Token Cost Anomaly | `sum:tau2.llm.token_cost{*}` | >2x avg | Alert |
| DR-004 | Premature Termination | `count:tau2.termination.reason{reason:max_errors}` | >10 in 1h | Create Incident |
| DR-005 | Latency SLO Breach | `p99:tau2.task.duration{*}` | >60s | SLO Alert |

**⭐ Hero Monitor**: DR-002 is the primary demo monitor - traffic generator prioritizes triggering low-reward scenarios to demonstrate the full Case creation workflow for judges.

### FR-004: Case/Incident Management
- Warning-level alerts SHALL create Cases in Datadog Case Management
- Critical-level alerts SHALL create Incidents
- Cases/Incidents SHALL include context:
  - `task_id` - The failing task identifier
  - `domain` - Evaluation domain (airline, retail, telecom)
  - `termination_reason` - Why the simulation ended
  - `error_details` - Relevant error messages
  - `runbook_link` - Link to investigation runbook
- **MVP Scope**: Manual classification only - Cases provide context for human triage
- **Deferred**: Automated `classify_failure()` routing to fix targets (see metrics_design.md)

### FR-005: Dashboard
- Dashboard SHALL show application health at a glance
- Required widgets:
  - Pass rate by domain (bar chart)
  - Average reward trend (time series)
  - Token usage and cost (time series)
  - Error rate (time series)
  - Termination reason breakdown (pie chart)
  - Latency percentiles (time series)
  - Active cases and incidents (query value)
  - SLO status (SLO widget)

### FR-006: Traffic Generator
- Script SHALL run tau2 evaluations to generate telemetry
- Script SHALL support configurable domains, task counts, concurrency
- Script SHALL demonstrate detection rule triggering
- Script SHALL include a "failure mode" that intentionally triggers detection rules:
  - Invalid tool call patterns (triggers MAX_ERRORS termination)
  - High-latency scenarios (triggers Latency SLO breach)
  - Forced low-reward tasks (triggers Task Failure Spike monitor)

### FR-007: Error Handling for External Dependencies
- On Gemini API failure: Log error prominently, skip current evaluation, continue with next
- On Datadog API failure: Log error prominently, continue evaluation (telemetry loss acceptable)
- All errors SHALL be visible in console output for demo transparency
- No hard stops - demo must remain resilient to transient API issues

## Deployment Constraints (from 008-gcp-integration)

### Cloud Run Limits

**Cloud Run imposes a 60-minute maximum request timeout.** This limits which benchmarks can be run synchronously.

| Domain | Total Tasks | Time per Task | Full Domain Time | Supported? |
|--------|-------------|---------------|------------------|------------|
| Mock | 9 | ~40s | ~6 min | ✅ Full |
| Airline | 50 | ~64s avg | ~53 min | ⚠️ Risky (borderline) |
| Retail | 114 | ~69s avg | ~2-3 hours | ❌ Task limit required |
| Telecom | 2,285 | ~191s avg | ~20+ days | ❌ Task limit required |

### Enforced Limits for Demo

| Parameter | Limit | Rationale |
|-----------|-------|-----------|
| `num_tasks` | Max 30 | ~30-40 min execution, safe margin |
| `num_trials` | Max 3 | Multiplies execution time |

### Supported Demo Use Cases

| Use Case | Supported | Notes |
|----------|-----------|-------|
| Quick smoke test (1-5 tasks) | ✅ | Any domain |
| Mock domain (full) | ✅ | 9 tasks, ~6 min |
| Airline sample (10-30 tasks) | ✅ | Subset of 50 |
| Retail sample (10-30 tasks) | ✅ | Subset of 114 |
| Telecom sample (10-30 tasks) | ✅ | Subset of 2,285 |
| Full domain evaluation | ❌ | Requires local execution |

## Technical Design

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install tau2-bench-agent from Git with ddtrace branch
RUN pip install git+https://github.com/wuTims/tau2-bench-agent.git@feature/ddtrace-instrumentation

# Install additional dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY datadog/ ./datadog/

# Set environment variables (aligned with 008-gcp-integration)
ENV DD_TRACE_ENABLED=true
ENV DD_LLMOBS_ENABLED=true
ENV DD_LLMOBS_AGENTLESS_ENABLED=true
ENV DD_SERVICE=tau2-datadog-observability
ENV DD_ENV=dev
ENV TAU2_AGENT_MODEL=gemini-2.0-flash
ENV PORT=8001

# Run with ddtrace
CMD ["ddtrace-run", "python", "-m", "scripts.traffic_generator"]
```

### Environment Variables (from 008-gcp-integration)

```bash
# Server-side LLM configuration
TAU2_AGENT_MODEL=gemini-2.0-flash          # Model for tau2_agent orchestrator
GOOGLE_API_KEY=AIza...                      # API key for Gemini (both orchestrator and user sim)

# Datadog configuration
DD_TRACE_ENABLED=true
DD_LLMOBS_ENABLED=true
DD_SERVICE=tau2-datadog-observability
DD_ENV=dev
DD_API_KEY=...                              # Datadog API key

# Application
LOG_LEVEL=INFO
PORT=8001
```

### Traffic Generator

```python
# scripts/traffic_generator.py
"""
Traffic generator for tau2-bench Datadog observability demo.

This script runs tau2 benchmark evaluations to generate telemetry
for Datadog LLM Observability, demonstrating detection rules.
"""

import os
import time
import subprocess
from datadog import initialize, statsd

# Initialize Datadog metrics
initialize(api_key=os.getenv("DD_API_KEY"))

DOMAINS = ["airline", "retail", "telecom"]
TASKS_PER_DOMAIN = 5
TRIALS = 1

def run_evaluation(domain: str, num_tasks: int = 5) -> dict:
    """Run tau2 evaluation and return results."""
    # Aligned with 008-gcp-integration: use gemini-2.0-flash for both LLMs
    cmd = [
        "tau2", "run",
        "--domain", domain,
        "--agent-llm", "gemini/gemini-2.0-flash",
        "--user-llm", "gemini/gemini-2.0-flash",
        "--num-tasks", str(num_tasks),
        "--num-trials", str(TRIALS),
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start

    # Emit custom metrics
    statsd.gauge("tau2.evaluation.duration", duration, tags=[f"domain:{domain}"])

    return {
        "domain": domain,
        "duration": duration,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

def main():
    """Run continuous evaluation loop for demo."""
    print("Starting tau2-bench traffic generator for Datadog demo...")

    while True:
        for domain in DOMAINS:
            print(f"Running evaluation: domain={domain}")
            result = run_evaluation(domain, TASKS_PER_DOMAIN)

            if result["returncode"] != 0:
                print(f"Evaluation failed: {result['stderr']}")
            else:
                print(f"Evaluation completed: {domain} in {result['duration']:.2f}s")

        # Wait between evaluation cycles
        time.sleep(60)

if __name__ == "__main__":
    main()
```

### Monitor Definition Example

```json
{
  "name": "tau2-bench: High Task Error Rate",
  "type": "metric alert",
  "query": "sum(last_5m):sum:tau2.task.error_count{env:production} / sum:tau2.task.total{env:production} > 0.2",
  "message": "{{#is_alert}}\nTask error rate exceeded 20% in the last 5 minutes.\n\n**Domain**: {{domain.name}}\n**Current Rate**: {{value}}%\n\n**Recommended Actions**:\n1. Check recent model changes\n2. Review error logs in APM\n3. Verify Gemini API health\n\n@slack-ai-alerts\n{{/is_alert}}",
  "tags": [
    "team:ai-platform",
    "service:tau2-bench",
    "env:production"
  ],
  "options": {
    "thresholds": {
      "critical": 0.2,
      "warning": 0.1
    },
    "notify_no_data": false,
    "renotify_interval": 60,
    "escalation_message": "Error rate still elevated after 1 hour. Creating incident."
  }
}
```

### SLO Definition Example

```json
{
  "name": "tau2-bench Task Success Rate SLO",
  "description": "99% of benchmark tasks should complete successfully",
  "type": "metric",
  "query": {
    "numerator": "sum:tau2.task.success{env:production}.as_count()",
    "denominator": "sum:tau2.task.total{env:production}.as_count()"
  },
  "target_threshold": 99.0,
  "warning_threshold": 99.5,
  "timeframe": "30d",
  "tags": [
    "team:ai-platform",
    "service:tau2-bench"
  ]
}
```

## Implementation Phases

### Phase 1: tau2-bench-agent ddtrace Integration (2-3 hours)

**Changes to tau2-bench-agent (zero core module modifications):**

1. Add `ddtrace>=4.0.0` and `datadog>=0.50.0` to `pyproject.toml`
2. Create `src/tau2/tracing.py` with configuration module
3. Create `src/experiments/datadog/scripts/tau2_traced.py` wrapper script
4. Test with `python -m experiments.datadog.scripts.tau2_traced run --domain mock`

**Note**: Uses wrapper-based approach - no modifications to `cli.py` or other core modules.

### Phase 2: New Repository Setup (2-3 hours)

1. Create `tau2-datadog-observability` repository
2. Set up project structure (Dockerfile, requirements.txt, etc.)
3. Write traffic generator script
4. Create README with deployment instructions

### Phase 3: GCP Deployment (2-3 hours)

1. Create GCP project
2. Enable Vertex AI APIs
3. Configure Cloud Run service
4. Set up service account with Vertex AI permissions
5. Deploy and verify Gemini calls work

### Phase 4: Datadog Configuration (3-4 hours)

1. Create Datadog trial account (or use extended trial from webinar)
2. Configure DD Agent / agentless mode
3. Verify LLM Observability traces appear
4. Create 5 monitors (detection rules)
5. Create 1 SLO
6. Create dashboard

### Phase 4.5: EvaluationStore Integration (ADR-002)

**Purpose**: Enable post-hoc metrics emission by persisting evaluation results to `$TAU2_DATA_DIR/evaluations/`.

**Why Before Phase 5**: Post-hoc custom metrics (`tau2.task.reward`, `tau2.evaluation.pass_rate`, etc.) are required for detection rules. Without EvaluationStore integration, `emit_metrics.py` cannot find evaluation data.

1. Update `RunTau2Evaluation._execute_streaming()` to call `store.create_session()` at start
2. Update `RunTau2Evaluation._execute()` to return full simulation data (not just summary)
3. Call `store.complete_evaluation()` with results after evaluation completes
4. Handle failures with `store.fail_evaluation()`
5. Verify `emit_metrics.py --all` finds and processes persisted evaluations

**Checkpoint**: Run evaluation via A2A, verify files appear in `$TAU2_DATA_DIR/evaluations/`, verify `emit_metrics.py` successfully emits metrics.

### Phase 5: Case Management & Documentation (2-3 hours)

1. Configure Case Management workflow
2. Set up Incident Management integration
3. Export all configurations to JSON
4. Write comprehensive README
5. Create demo video / screenshots

## Submission Checklist

| Requirement | Deliverable | Status |
|-------------|-------------|--------|
| Hosted application URL | Cloud Run URL | Pending (Deferred to 008-gcp-integration) |
| Public repo with OSI license | GitHub repo with Apache-2.0 | ✅ Complete (`src/experiments/datadog/LICENSE`) |
| Instrumented LLM application | tau2-bench + ddtrace | ✅ Complete (`src/tau2/tracing.py`) |
| README with deployment instructions | README.md | ✅ Complete (`src/experiments/datadog/README.md`) |
| JSON export of Datadog configs | `datadog/*.json` | ✅ Complete (`src/experiments/datadog/configs/`) |
| Datadog organization name | To be created | Pending (requires DD account) |
| Traffic generator script | `scripts/traffic_generator.py` | ✅ Complete (`src/experiments/datadog/scripts/traffic_generator.py`) |
| Vertex AI / Gemini usage | LiteLLM config | ✅ Complete (via Nebius/OpenAI-compatible API) |
| 3+ detection rules | 5 monitors defined | ✅ Complete (DR-001 to DR-005 in `monitors.json`) |
| Actionable records (Case/Incident) | Case Management workflow | ✅ Complete (`case_templates.json` + monitor configs) |
| Dashboard showing app health | Dashboard JSON | ✅ Complete (`dashboards.json`) |

## What This Project IS

- A **wrapper/deployment** around tau2-bench-agent
- An **observability demonstration** for the hackathon
- A **reference implementation** for LLM observability patterns

## What This Project is NOT

- NOT a modification of tau2-bench-agent's core functionality
- NOT a new benchmarking framework
- NOT a replacement for tau2's existing features

## Success Criteria

- **SC-001**: Gemini LLM calls appear in Datadog LLM Observability
- **SC-002**: Custom tau2 metrics visible in Datadog Metrics Explorer
- **SC-003**: At least one detection rule triggers during demo
- **SC-004**: Case or Incident automatically created from alert
- **SC-005**: Dashboard loads and shows real-time data
- **SC-006**: End-to-end demo runs without manual intervention

## Clarifications

### Session 2025-12-24
- Q: Which Datadog deployment mode for LLM Observability in Cloud Run? → A: Agentless mode
- Q: Should demo include intentional failure scenarios to trigger detection rules? → A: Yes, include scripted failure scenarios
- Q: How should the system handle external dependency failures (Gemini/Datadog API)? → A: Log and continue - log errors prominently but continue running other evaluations
- Q: Is automated failure classification (classify_failure) in scope for hackathon MVP? → A: No, manual classification only - Cases include context but humans determine fix target
- Q: Which detection rule should be the "hero" for demo? → A: DR-002 Task Failure Spike (avg reward <0.7) - most relevant to LLM quality

## Open Questions

1. **Datadog Trial Duration**: Need to confirm 14-day trial is sufficient or attend Dec 9 webinar for +30 days
2. ~~**GCP Billing**: Vertex AI has free tier limits~~ → **RESOLVED**: Using Gemini Developer API (not Vertex AI) per 008-gcp-integration. Gemini 2.0 Flash is cost-effective.
3. ~~**Agent vs Agentless**: Which mode for LLM Observability in Cloud Run?~~ → **RESOLVED**: Agentless mode - sends data directly to Datadog intake API without sidecar, ideal for Cloud Run's ephemeral container model.
4. ~~**Demo Scenario**: Should we intentionally trigger failures to demonstrate detection rules?~~ → **RESOLVED**: Yes, include scripted failure scenarios (invalid tool calls, timeout triggers) to demonstrate detection rules and Case/Incident creation for judges.

## Alignment with 008-gcp-integration

This spec is aligned with the GCP Integration spec (008) for:

| Decision | Value | Rationale |
|----------|-------|-----------|
| LLM Provider | Gemini Developer API | Cost-effective, no Vertex AI setup needed |
| Model | `gemini-2.0-flash` | Fast, cheap, sufficient for demo |
| Cloud Run timeout | 60 minutes | Platform constraint |
| Max tasks | 30 | Safe margin for timeout |
| BYOK | Not used (demo) | Server pays for hackathon demo |

## References

- [Datadog LLM Observability](https://docs.datadoghq.com/llm_observability/)
- [Monitor Google Gemini with Datadog](https://www.datadoghq.com/blog/monitor-google-gemini-datadog-llm-observability/)
- [Datadog Case Management](https://docs.datadoghq.com/service_management/case_management/)
- [Datadog Monitor-based SLOs](https://docs.datadoghq.com/service_management/service_level_objectives/monitor/)
- [dd-trace-py LLM Observability](https://ddtrace.readthedocs.io/en/stable/integrations.html#llm-observability)
- [tau2-bench-agent DeepWiki](https://deepwiki.com/wuTims/tau2-bench-agent)
