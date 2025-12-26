# Data Model: Datadog LLM Observability Hackathon Project

**Feature Branch**: `007-datadog-project`
**Date**: 2025-12-24

## Overview

This data model defines the metrics, logs, and configuration entities for the Datadog observability integration. The model is derived from [metrics_design.md](metrics_design.md) and aligns with Datadog's native schemas.

## Entities

### 1. TaskMetrics

Represents metrics emitted for each tau2 task evaluation.

| Field | Type | Tags | Description |
|-------|------|------|-------------|
| `tau2.task.reward` | gauge | task_id, domain, evaluation_id | Task reward (0.0-1.0) |
| `tau2.task.steps` | gauge | task_id, domain | Steps taken in task |
| `tau2.task.duration_seconds` | histogram | task_id, domain | Task execution time |
| `tau2.task.success` | count | task_id, domain, success:bool | Task success (reward >= 0.7) / failure |
| `tau2.task.total` | count | domain, evaluation_id | Total tasks evaluated (for ratio calculations) |

**Source**: `simulation.reward_info.reward`, `simulation.duration`, `len(simulation.messages)`

**Derived Metrics** (for monitors):
- Error rate = `sum:tau2.task.success{success:false} / sum:tau2.task.total`
- Pass rate = `sum:tau2.task.success{success:true} / sum:tau2.task.total * 100`

### 2. ToolMetrics

Represents metrics for tool invocations during evaluation.

| Field | Type | Tags | Description |
|-------|------|------|-------------|
| `tau2.tool.calls` | count | tool_name, requestor, domain | Tool invocations |
| `tau2.tool.correct` | count | tool_name, correct:bool | Tool call correctness |
| `tau2.tool.arguments_match` | count | tool_name, match:bool | Argument correctness |

**Source**: `simulation.messages[].tool_calls`, `simulation.reward_info.action_checks`

### 3. AssertionMetrics

Represents metrics for assertion evaluations.

| Field | Type | Tags | Description |
|-------|------|------|-------------|
| `tau2.assertion.result` | count | type, met:bool | Assertion pass/fail |
| `tau2.assertion.nl_failed` | count | assertion_text, task_id | NL assertion failures |

**Assertion Types**: `db`, `action`, `nl`, `communicate`

**Source**: `simulation.reward_info.{db_check, action_checks, nl_assertions, communicate_checks}`

### 4. TerminationMetrics

Represents why tasks ended.

| Field | Type | Tags | Description |
|-------|------|------|-------------|
| `tau2.termination` | count | reason | Why tasks ended |

**Termination Reasons**: `user_stop`, `agent_stop`, `max_steps`, `max_errors`

**Source**: `simulation.termination_reason`

### 5. EvaluationMetrics

Aggregated metrics for entire evaluation runs.

| Field | Type | Tags | Description |
|-------|------|------|-------------|
| `tau2.evaluation.pass_rate` | gauge | domain, evaluation_id | Overall pass rate (%) |
| `tau2.evaluation.avg_reward` | gauge | domain | Average reward |
| `tau2.evaluation.tasks_total` | gauge | domain | Total tasks in evaluation |

**Source**: `evaluation.results.{success_rate, avg_reward}`

### 6. LLM Metrics (Auto-Instrumented)

These metrics are captured automatically by `ddtrace.patch(litellm=True)` via LLM Observability. **No custom emission required.**

| Field | Type | Tags | Description |
|-------|------|------|-------------|
| `tau2.llm.tokens_input` | count | model, domain | Input tokens per LLM call |
| `tau2.llm.tokens_output` | count | model, domain | Output tokens per LLM call |
| `tau2.llm.token_cost` | gauge | model, domain | Estimated cost per LLM call |
| `tau2.llm.latency` | histogram | model, domain | LLM response latency |

**Source**: ddtrace LiteLLM auto-instrumentation (LLM Observability)

**Note**: Token cost is calculated by ddtrace based on model pricing. For `gemini-2.0-flash`, costs are derived from Google's published pricing.

## Log Schema

### Base Fields (all logs)

```json
{
  "timestamp": "2025-12-24T10:00:00Z",
  "service": "tau2-bench-agent",
  "env": "production",
  "evaluation_id": "eval-1732449600000-a1b2c3",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "domain": "airline"
}
```

### Event: tau2.task.completed

```json
{
  "event": "tau2.task.completed",
  "task_id": "3",
  "reward": 1.0,
  "success": true,
  "termination_reason": "user_stop",
  "steps": 14,
  "duration_seconds": 29.3,
  "reward_breakdown": {
    "db": 1.0,
    "communicate": 1.0,
    "action": 1.0,
    "nl": 1.0
  }
}
```

### Event: tau2.task.failed

```json
{
  "event": "tau2.task.failed",
  "task_id": "7",
  "reward": 0.0,
  "termination_reason": "max_errors",
  "failure_reasons": [
    {
      "type": "action_check",
      "tool": "update_reservation",
      "message": "Incorrect passenger count"
    }
  ]
}
```

### Event: tau2.assertion.evaluated

```json
{
  "event": "tau2.assertion.evaluated",
  "task_id": "3",
  "assertion_type": "nl",
  "assertion_text": "Agent detects user is Silver member",
  "met": false,
  "justification": "Agent did not confirm membership status"
}
```

## Configuration Entities

### Monitor Definition

```json
{
  "name": "string",
  "type": "metric alert | log alert",
  "query": "string",
  "message": "string (markdown with template vars)",
  "tags": ["string"],
  "options": {
    "thresholds": {
      "critical": "number",
      "warning": "number"
    },
    "notify_no_data": "boolean",
    "renotify_interval": "number (minutes)"
  }
}
```

### SLO Definition

```json
{
  "name": "string",
  "description": "string",
  "type": "metric",
  "query": {
    "numerator": "string",
    "denominator": "string"
  },
  "target_threshold": "number (0-100)",
  "warning_threshold": "number (0-100)",
  "timeframe": "string (7d, 30d, etc.)",
  "tags": ["string"]
}
```

### Dashboard Widget

```json
{
  "title": "string",
  "type": "timeseries | heatmap | toplist | sunburst | query_value | log_stream",
  "requests": [
    {
      "q": "string (metric query)",
      "display_type": "line | bars | area"
    }
  ]
}
```

### Case Template

```yaml
title: "string (with {{template_vars}})"
priority: "P1 | P2 | P3 | P4"
team: "string"
labels:
  - "string"
context:
  key: "value"
links:
  - title: "string"
    url: "string"
runbook: "string (markdown)"
```

## State Transitions

### Task Lifecycle

```
STARTED
    │
    ▼
RUNNING ───► tool.called ───► tool.result
    │                              │
    │◄─────────────────────────────┘
    │
    ▼
EVALUATING ───► assertion.evaluated (×N)
    │
    ├──► COMPLETED (reward >= 0.7)
    │
    └──► FAILED (reward < 0.7 or termination=max_errors)
```

### Monitor → Case Workflow

```
MONITOR_TRIGGERED
    │
    ├─── warning threshold ───► CREATE_CASE
    │
    └─── critical threshold ───► CREATE_INCIDENT
```

## Validation Rules

### Metric Tags

| Tag | Validation | Example |
|-----|------------|---------|
| `task_id` | Non-empty string | `"3"` |
| `domain` | One of: airline, retail, telecom, mock | `"airline"` |
| `evaluation_id` | Pattern: `eval-{timestamp}-{hash}` | `"eval-1732449600000-a1b2c3"` |
| `success` | Boolean string | `"true"` or `"false"` |
| `reason` | One of: user_stop, agent_stop, max_steps, max_errors | `"max_errors"` |

### Log Fields

| Field | Validation |
|-------|------------|
| `reward` | Float in range [0.0, 1.0] |
| `duration_seconds` | Positive float |
| `steps` | Positive integer |
| `trace_id` | 32-character hex string |

## Relationships

```
Evaluation (1) ─────► (N) Task
     │
     └─── evaluation_id links metrics

Task (1) ─────► (N) ToolCall
     │
     └─── task_id links tool metrics

Task (1) ─────► (N) Assertion
     │
     └─── task_id links assertion metrics

Monitor (1) ─────► (N) Case/Incident
     │
     └─── monitor triggers cases

Dashboard (1) ─────► (N) Widget
     │
     └─── dashboard contains widgets
```
