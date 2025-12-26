# Datadog Metrics Design: Post-Hoc Evaluation Analysis

## Overview

This document defines the metrics, logs, and detection rules for tau2-bench observability in Datadog. The design enables:

1. **Task-level failure tracking** - Identify exactly which tasks failed and why
2. **Root cause analysis** - Correlate tool calls, assertions, and conversation patterns
3. **Actionable remediation** - Cases with specific context and next steps

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  tau2 Evaluation Completes                                          │
│                                                                      │
│  data/evaluations/eval-{id}.json                                    │
│  ├── messages[]           ← Full conversation                       │
│  ├── reward_info          ← Detailed reward breakdown               │
│  │   ├── db_check         ← Database state verification             │
│  │   ├── action_checks[]  ← Tool call correctness                   │
│  │   ├── nl_assertions[]  ← Natural language verification           │
│  │   └── communicate_checks[] ← Required info communicated          │
│  └── termination_reason   ← Why simulation ended                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Metrics Emitter (runs after evaluation)                            │
│                                                                      │
│  Reads JSON → Emits to Datadog:                                     │
│  • Custom metrics (statsd)                                          │
│  • Structured logs (with trace_id)                                  │
│  • Events (for significant failures)                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Custom Metrics

### Task-Level Metrics

| Metric Name | Type | Tags | Description |
|-------------|------|------|-------------|
| `tau2.task.reward` | gauge | `task_id`, `domain`, `evaluation_id` | Task reward (0.0-1.0) |
| `tau2.task.steps` | gauge | `task_id`, `domain` | Steps taken in task |
| `tau2.task.duration_seconds` | histogram | `task_id`, `domain` | Task execution time |
| `tau2.task.success` | count | `task_id`, `domain`, `success:true/false` | Task success/failure |

### Tool-Level Metrics

| Metric Name | Type | Tags | Description |
|-------------|------|------|-------------|
| `tau2.tool.calls` | count | `tool_name`, `requestor:agent/user`, `domain` | Tool invocations |
| `tau2.tool.correct` | count | `tool_name`, `correct:true/false` | Tool call correctness per action_checks |
| `tau2.tool.arguments_match` | count | `tool_name`, `match:true/false` | Argument correctness |

### Assertion Metrics

| Metric Name | Type | Tags | Description |
|-------------|------|------|-------------|
| `tau2.assertion.result` | count | `type:db/nl/action/communicate`, `met:true/false` | Assertion pass/fail |
| `tau2.assertion.nl_failed` | count | `assertion_text`, `task_id` | Specific NL assertion failures |

### Termination Metrics

| Metric Name | Type | Tags | Description |
|-------------|------|------|-------------|
| `tau2.termination` | count | `reason:user_stop/agent_stop/max_steps/max_errors` | Why tasks ended |

### Evaluation-Level Metrics

| Metric Name | Type | Tags | Description |
|-------------|------|------|-------------|
| `tau2.evaluation.pass_rate` | gauge | `domain`, `evaluation_id` | Overall pass rate |
| `tau2.evaluation.avg_reward` | gauge | `domain` | Average reward across tasks |
| `tau2.evaluation.tasks_total` | gauge | `domain` | Total tasks in evaluation |

## Structured Logs

### Log Schema

All logs include these base fields for correlation:

```json
{
  "timestamp": "2025-12-23T10:00:00Z",
  "service": "tau2-bench-agent",
  "env": "production",
  "evaluation_id": "eval-1732449600000-a1b2c3",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "domain": "airline"
}
```

### Event Types

#### `tau2.task.started`
```json
{
  "event": "tau2.task.started",
  "task_id": "3",
  "task_purpose": "Check baggage allowance for silver member"
}
```

#### `tau2.tool.called`
```json
{
  "event": "tau2.tool.called",
  "task_id": "3",
  "tool_name": "get_reservation_details",
  "requestor": "agent",
  "arguments": {"reservation_id": "JMO1MG"},
  "turn_idx": 4
}
```

#### `tau2.tool.result`
```json
{
  "event": "tau2.tool.result",
  "task_id": "3",
  "tool_name": "get_reservation_details",
  "success": true,
  "result_length": 1234
}
```

#### `tau2.assertion.evaluated`
```json
{
  "event": "tau2.assertion.evaluated",
  "task_id": "3",
  "assertion_type": "nl",
  "assertion_text": "Agent detects that user is actually a Silver member",
  "met": true,
  "justification": "The agent confirmed the user's membership status as Silver..."
}
```

#### `tau2.task.completed`
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

#### `tau2.task.failed`
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
      "expected_args": {"passengers": 2},
      "actual_args": {"passengers": 3},
      "message": "Incorrect passenger count in reservation update"
    },
    {
      "type": "nl_assertion",
      "assertion": "Agent correctly applies discount",
      "met": false,
      "justification": "Agent did not mention the applicable discount"
    }
  ]
}
```

## Detection Rules

### Monitor 1: Task Failure Rate by Domain

**Query**:
```
sum(last_10m):sum:tau2.task.success{success:false} by {domain}.as_count() /
sum:tau2.task.success{*} by {domain}.as_count() > 0.3
```

**Alert Message**:
```
{{#is_alert}}
🚨 **High Task Failure Rate in {{domain.name}}**

**Current Failure Rate**: {{value}}%
**Threshold**: 30%

**Recent Failures** (from logs):
{{#logs}}
- Task {{task_id}}: {{termination_reason}} (reward: {{reward}})
{{/logs}}

**Recommended Actions**:
1. Check if domain tools have changed
2. Review recent agent prompt modifications
3. Verify user simulator behavior

@slack-tau2-alerts
{{/is_alert}}
```

### Monitor 2: Specific Tool Failure Pattern

**Query** (Log-based):
```
logs("service:tau2-bench-agent event:tau2.tool.called")
.rollup("count").by("tool_name").last("10m") > 10
AND
logs("service:tau2-bench-agent event:tau2.assertion.evaluated met:false")
.rollup("count").last("10m") > 5
```

**Alert Message**:
```
{{#is_alert}}
🔧 **Tool Usage Pattern Issue Detected**

The agent is making many tool calls but still failing assertions.

**Top Failed Tools**:
{{#facets.tool_name}}
- {{name}}: {{count}} calls
{{/facets.tool_name}}

**Failed Assertions**:
{{#logs}}
- {{assertion_text}}: {{justification}}
{{/logs}}

**Remediation**:
1. Agent may be confused about tool usage
2. Check if tool responses match agent expectations
3. Review domain policy for clarity

{{/is_alert}}
```

### Monitor 3: Termination Reason Anomaly

**Query**:
```
sum(last_1h):sum:tau2.termination{reason:max_errors}.as_count() > 5
```

**Alert Message**:
```
{{#is_alert}}
⚠️ **Excessive MAX_ERRORS Terminations**

Tasks are hitting the error limit and being terminated prematurely.

**Count**: {{value}} tasks in the last hour
**Threshold**: 5

**This indicates**:
- Agent making invalid tool calls
- Tool schema mismatch
- Environment state corruption

**Immediate Actions**:
1. Check latest tool execution errors in logs
2. Verify tool schemas haven't changed
3. Review agent's tool call patterns

@pagerduty-tau2-critical
{{/is_alert}}
```

### Monitor 4: NL Assertion Failure Pattern

**Query** (Log-based):
```
logs("service:tau2-bench-agent event:tau2.assertion.evaluated assertion_type:nl met:false")
.rollup("count").by("assertion_text").last("30m") > 3
```

**Alert Message**:
```
{{#is_alert}}
📝 **Repeated NL Assertion Failure**

The same natural language assertion is failing across multiple tasks.

**Failing Assertion**: "{{assertion_text}}"
**Failure Count**: {{value}} in last 30 minutes

**Sample Justifications**:
{{#logs}}
- Task {{task_id}}: "{{justification}}"
{{/logs}}

**This suggests**:
- Agent not communicating required information
- Prompt may be missing key instructions
- User simulator not eliciting needed response

**Remediation**:
1. Review agent system prompt for this domain
2. Check if required info is in agent's knowledge
3. Analyze conversation flow in failed tasks

{{/is_alert}}
```

### Monitor 5: Low Average Reward SLO

**SLO Definition**:
```json
{
  "name": "tau2-bench Task Quality SLO",
  "description": "95% of tasks should achieve reward >= 0.7",
  "type": "metric",
  "query": {
    "numerator": "sum:tau2.task.reward{reward:>=0.7}.as_count()",
    "denominator": "sum:tau2.task.reward{*}.as_count()"
  },
  "target_threshold": 95.0,
  "warning_threshold": 97.0,
  "timeframe": "7d"
}
```

## Case Templates

### Template 1: Task Failure Investigation

**Trigger**: Monitor 1 or Monitor 3 fires

**Case Fields**:
```yaml
title: "Task Failure: {{task_id}} in {{domain}}"
priority: P2
team: ai-platform
labels:
  - tau2-bench
  - task-failure
  - {{domain}}

context:
  evaluation_id: "{{evaluation_id}}"
  task_id: "{{task_id}}"
  reward: "{{reward}}"
  termination_reason: "{{termination_reason}}"

links:
  - title: "View APM Trace"
    url: "https://app.datadoghq.com/apm/trace/{{trace_id}}"
  - title: "View Logs"
    url: "https://app.datadoghq.com/logs?query=evaluation_id:{{evaluation_id}}"
  - title: "Evaluation JSON"
    url: "{{storage_url}}/evaluations/{{evaluation_id}}.json"

runbook: |
  ## Investigation Steps

  ### 1. Check Reward Breakdown
  Look at the `reward_breakdown` in the evaluation JSON:
  - `db`: Did database state match expected?
  - `action`: Were tool calls correct?
  - `nl`: Were NL assertions met?
  - `communicate`: Was required info communicated?

  ### 2. Review Conversation Flow
  Open the APM trace and examine:
  - User simulator prompts (what did user ask?)
  - Agent responses (did agent understand?)
  - Tool calls (were they appropriate?)

  ### 3. Compare to Successful Task
  Find a similar successful task in the same domain and compare:
  - Tool call sequence
  - Information communicated
  - Response patterns

  ### 4. Check for Systemic Issues
  - Are multiple tasks failing with same pattern?
  - Is this domain-specific?
  - Did a recent deployment cause this?
```

### Template 2: Tool Pattern Issue

**Trigger**: Monitor 2 fires

**Case Fields**:
```yaml
title: "Tool Usage Issue: {{tool_name}} in {{domain}}"
priority: P3
team: ai-platform

context:
  tool_name: "{{tool_name}}"
  call_count: "{{call_count}}"
  failure_rate: "{{failure_rate}}"

runbook: |
  ## Tool Issue Investigation

  ### 1. Analyze Tool Call Patterns
  Query logs for this tool:
  ```
  service:tau2-bench-agent event:tau2.tool.called tool_name:{{tool_name}}
  ```

  Look for:
  - Repeated calls with same arguments (agent stuck in loop)
  - Invalid argument patterns
  - Calls at wrong point in conversation

  ### 2. Check Tool Schema
  - Has the tool schema changed recently?
  - Are arguments matching expected types?
  - Is the tool returning expected format?

  ### 3. Review Agent Prompt
  - Does the system prompt explain when to use this tool?
  - Are tool usage examples included?
  - Is the tool description clear?
```

## Dashboard Widgets

### Widget 1: Task Success Rate by Domain (Timeseries)

```json
{
  "title": "Task Success Rate by Domain",
  "type": "timeseries",
  "requests": [
    {
      "q": "sum:tau2.task.success{success:true} by {domain}.as_rate() / sum:tau2.task.success{*} by {domain}.as_rate() * 100",
      "display_type": "line"
    }
  ]
}
```

### Widget 2: Reward Distribution (Heatmap)

```json
{
  "title": "Task Reward Distribution",
  "type": "heatmap",
  "requests": [
    {
      "q": "avg:tau2.task.reward{*} by {task_id,domain}"
    }
  ]
}
```

### Widget 3: Tool Call Breakdown (Top List)

```json
{
  "title": "Tool Calls by Type",
  "type": "toplist",
  "requests": [
    {
      "q": "sum:tau2.tool.calls{*} by {tool_name,requestor}.as_count()",
      "conditional_formats": [
        {"comparator": ">", "value": 100, "palette": "green_on_white"},
        {"comparator": ">", "value": 50, "palette": "yellow_on_white"}
      ]
    }
  ]
}
```

### Widget 4: Failure Reasons (Pie Chart)

```json
{
  "title": "Task Termination Reasons",
  "type": "sunburst",
  "requests": [
    {
      "q": "sum:tau2.termination{*} by {reason}.as_count()"
    }
  ]
}
```

### Widget 5: Assertion Pass Rate (Query Value)

```json
{
  "title": "NL Assertion Pass Rate",
  "type": "query_value",
  "requests": [
    {
      "q": "sum:tau2.assertion.result{type:nl,met:true}.as_count() / sum:tau2.assertion.result{type:nl}.as_count() * 100",
      "conditional_formats": [
        {"comparator": ">=", "value": 90, "palette": "green_on_white"},
        {"comparator": ">=", "value": 70, "palette": "yellow_on_white"},
        {"comparator": "<", "value": 70, "palette": "red_on_white"}
      ]
    }
  ],
  "precision": 1
}
```

### Widget 6: Recent Failed Tasks (Log Stream)

```json
{
  "title": "Recent Task Failures",
  "type": "log_stream",
  "query": "service:tau2-bench-agent event:tau2.task.failed",
  "columns": ["task_id", "domain", "reward", "termination_reason"],
  "message_display": "expanded-md"
}
```

## Implementation: Python Metrics Emitter

```python
# scripts/emit_metrics.py
"""
Emit Datadog metrics from stored tau2 evaluation results.

Reads JSON evaluation files and emits:
- Custom metrics via DogStatsD
- Structured logs via Datadog Logs API
- Events for significant failures
"""

import json
import os
from pathlib import Path
from datadog import initialize, statsd, api
from datetime import datetime

# Initialize Datadog
initialize(api_key=os.getenv("DD_API_KEY"))

EVALUATIONS_DIR = Path(os.getenv("TAU2_DATA_DIR", "./data")) / "evaluations"


def emit_task_metrics(simulation: dict, evaluation_id: str, domain: str, trace_id: str):
    """Emit metrics for a single task/simulation."""
    task_id = simulation.get("task_id", "unknown")
    reward = simulation.get("reward_info", {}).get("reward", 0.0)
    termination_reason = simulation.get("termination_reason", "unknown")
    duration = simulation.get("duration", 0.0)
    messages = simulation.get("messages", [])

    tags = [
        f"task_id:{task_id}",
        f"domain:{domain}",
        f"evaluation_id:{evaluation_id}",
    ]

    # Task-level metrics
    statsd.gauge("tau2.task.reward", reward, tags=tags)
    statsd.gauge("tau2.task.steps", len(messages), tags=tags)
    statsd.histogram("tau2.task.duration_seconds", duration, tags=tags)
    statsd.increment("tau2.task.success", tags=tags + [f"success:{reward >= 0.7}"])
    statsd.increment("tau2.termination", tags=tags + [f"reason:{termination_reason}"])

    # Tool-level metrics
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_tags = tags + [
                    f"tool_name:{tc.get('name', 'unknown')}",
                    f"requestor:agent",
                ]
                statsd.increment("tau2.tool.calls", tags=tool_tags)

        if msg.get("role") == "user" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_tags = tags + [
                    f"tool_name:{tc.get('name', 'unknown')}",
                    f"requestor:user",
                ]
                statsd.increment("tau2.tool.calls", tags=tool_tags)

    # Assertion metrics
    reward_info = simulation.get("reward_info", {})

    # Action checks
    for check in reward_info.get("action_checks", []):
        action = check.get("action", {})
        statsd.increment("tau2.assertion.result", tags=tags + [
            "type:action",
            f"met:{check.get('action_match', False)}",
            f"tool_name:{action.get('name', 'unknown')}",
        ])

    # NL assertions
    for assertion in reward_info.get("nl_assertions", []):
        statsd.increment("tau2.assertion.result", tags=tags + [
            "type:nl",
            f"met:{assertion.get('met', False)}",
        ])

        # Emit log for failed NL assertions
        if not assertion.get("met", True):
            emit_structured_log({
                "event": "tau2.assertion.evaluated",
                "task_id": task_id,
                "evaluation_id": evaluation_id,
                "trace_id": trace_id,
                "domain": domain,
                "assertion_type": "nl",
                "assertion_text": assertion.get("nl_assertion", ""),
                "met": False,
                "justification": assertion.get("justification", ""),
            })

    # Communicate checks
    for check in reward_info.get("communicate_checks", []):
        statsd.increment("tau2.assertion.result", tags=tags + [
            "type:communicate",
            f"met:{check.get('met', False)}",
        ])

    # DB check
    db_check = reward_info.get("db_check", {})
    if db_check:
        statsd.increment("tau2.assertion.result", tags=tags + [
            "type:db",
            f"met:{db_check.get('db_match', False)}",
        ])


def emit_structured_log(log_data: dict):
    """Emit structured log to Datadog."""
    log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    log_data["service"] = "tau2-bench-agent"
    log_data["env"] = os.getenv("DD_ENV", "production")

    # Use Datadog Logs API
    api.Logs.send(logs=[{
        "ddsource": "python",
        "ddtags": f"service:tau2-bench-agent,env:{log_data['env']}",
        "message": json.dumps(log_data),
    }])


def emit_failure_event(task_id: str, domain: str, failure_reasons: list):
    """Emit Datadog Event for significant failures."""
    api.Event.create(
        title=f"Task Failure: {task_id} in {domain}",
        text=f"Task failed with reasons:\n" + "\n".join(
            f"- {r['type']}: {r.get('message', r.get('assertion', 'Unknown'))}"
            for r in failure_reasons
        ),
        alert_type="error",
        tags=[f"domain:{domain}", f"task_id:{task_id}"],
    )


def process_evaluation(eval_path: Path):
    """Process a single evaluation file."""
    with open(eval_path) as f:
        data = json.load(f)

    evaluation_id = data.get("evaluation_id", eval_path.stem)
    trace_id = data.get("trace_id", "")
    domain = data.get("domain", "unknown")

    # Emit evaluation-level metrics
    results = data.get("results", {})
    statsd.gauge("tau2.evaluation.pass_rate",
                 results.get("success_rate", 0) * 100,
                 tags=[f"domain:{domain}", f"evaluation_id:{evaluation_id}"])
    statsd.gauge("tau2.evaluation.avg_reward",
                 results.get("avg_reward", 0),
                 tags=[f"domain:{domain}"])

    # Process each simulation/task
    for sim in data.get("simulations", []):
        emit_task_metrics(sim, evaluation_id, domain, trace_id)


def main():
    """Process all evaluation files."""
    for eval_file in EVALUATIONS_DIR.glob("eval-*.json"):
        print(f"Processing {eval_file.name}")
        process_evaluation(eval_file)

    print("Metrics emission complete")


if __name__ == "__main__":
    main()
```

## Failure Classification & Remediation Routing

The key insight is that different failure patterns indicate different root causes, requiring different fixes:

### Failure Pattern → Root Cause → Remediation Matrix

| Failure Signal | Likely Root Cause | Fix Target | Remediation |
|----------------|-------------------|------------|-------------|
| `db_check` failed, agent never called required tool | **Agent** doesn't know which tool to use | **A2A Agent** | Add tool usage examples to agent prompt |
| `db_check` failed, tool called with wrong args | **Agent** misunderstands tool schema | **A2A Agent** | Clarify tool parameter descriptions |
| `action_check` failed, correct tool but wrong order | **Agent** doesn't understand workflow | **A2A Agent** | Add step-by-step workflow to agent prompt |
| `nl_assertion` failed, info was in tool response | **Agent** didn't communicate info to user | **A2A Agent** | Add instruction to relay specific info |
| `nl_assertion` failed, info NOT in tool response | **Environment** missing data | **Environment** | Check environment state initialization |
| `communicate_check` failed | **Agent** didn't say required text | **A2A Agent** | Add explicit communication requirements |
| `termination:max_errors` | **Environment** rejecting valid calls | **Environment** | Check tool validation logic |
| `termination:max_steps` | **User Simulator** not providing info | **User Simulator** | Check user LLM prompt for persona |
| User LLM error in trace | **User Simulator** prompt issue | **User Simulator** | Fix user simulator system prompt |
| Task reward=0 but similar tasks pass | **Task** definition inconsistent | **Task** | Review task assertions vs expected behavior |

### Automated Failure Classification

```python
def classify_failure(simulation: dict) -> dict:
    """
    Classify task failure and determine remediation target.

    Returns:
        {
            "target": "agent" | "environment" | "user_simulator" | "task",
            "category": str,
            "evidence": list[str],
            "remediation": str
        }
    """
    reward_info = simulation.get("reward_info", {})
    termination = simulation.get("termination_reason", "")
    messages = simulation.get("messages", [])

    # Count tool calls by agent
    agent_tool_calls = [
        msg for msg in messages
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]

    # Check what failed
    db_check = reward_info.get("db_check", {})
    action_checks = reward_info.get("action_checks", [])
    nl_assertions = reward_info.get("nl_assertions", [])
    communicate_checks = reward_info.get("communicate_checks", [])

    failed_actions = [c for c in action_checks if not c.get("action_match")]
    failed_nl = [a for a in nl_assertions if not a.get("met")]
    failed_communicate = [c for c in communicate_checks if not c.get("met")]

    # Classification logic
    if termination == "max_errors":
        return {
            "target": "environment",
            "category": "tool_execution_errors",
            "evidence": ["Task hit max errors limit"],
            "remediation": "Check tool validation logic and error handling"
        }

    if termination == "max_steps" and len(agent_tool_calls) < 3:
        return {
            "target": "user_simulator",
            "category": "user_not_providing_info",
            "evidence": ["Task timed out", "Agent made few tool calls"],
            "remediation": "User simulator may not be providing necessary info"
        }

    if not db_check.get("db_match", True):
        if len(agent_tool_calls) == 0:
            return {
                "target": "agent",
                "category": "agent_no_tool_calls",
                "evidence": ["DB state wrong", "Agent never called tools"],
                "remediation": "Agent needs tool usage instructions"
            }
        else:
            return {
                "target": "agent",
                "category": "agent_wrong_tool_usage",
                "evidence": ["DB state wrong", f"Agent made {len(agent_tool_calls)} tool calls"],
                "remediation": "Agent using tools incorrectly"
            }

    if failed_actions:
        return {
            "target": "agent",
            "category": "incorrect_tool_calls",
            "evidence": [f"Failed action: {a['action']['name']}" for a in failed_actions],
            "remediation": "Agent needs clearer tool schemas or examples"
        }

    if failed_nl:
        # Check if info was available in tool responses
        tool_responses = [msg.get("content", "") for msg in messages if msg.get("role") == "tool"]
        all_tool_content = " ".join(str(r) for r in tool_responses)

        # Simple heuristic: if assertion mentions something in tool response
        for assertion in failed_nl:
            assertion_text = assertion.get("nl_assertion", "")
            if any(keyword in all_tool_content.lower() for keyword in assertion_text.lower().split()[:3]):
                return {
                    "target": "agent",
                    "category": "agent_didnt_communicate",
                    "evidence": [f"Assertion: {assertion_text}", "Info was in tool response"],
                    "remediation": "Agent has info but didn't communicate it to user"
                }

        return {
            "target": "environment",
            "category": "missing_data",
            "evidence": [f"Assertion: {a['nl_assertion']}" for a in failed_nl],
            "remediation": "Required info may not be in environment state"
        }

    if failed_communicate:
        return {
            "target": "agent",
            "category": "missing_communication",
            "evidence": [f"Required: {c.get('info')}" for c in failed_communicate],
            "remediation": "Agent must explicitly state required information"
        }

    # Default
    return {
        "target": "task",
        "category": "unknown",
        "evidence": ["No clear failure pattern"],
        "remediation": "Manual review required"
    }
```

### Case Templates by Fix Target

#### Case: Fix A2A Agent

**Trigger**: `classification.target == "agent"`

```yaml
title: "Agent Fix Required: {{category}} in task {{task_id}}"
priority: P2
team: ai-platform
labels:
  - agent-fix
  - {{domain}}
  - {{category}}

runbook: |
  ## Agent Behavior Issue

  **Category**: {{category}}
  **Evidence**:
  {{#evidence}}
  - {{.}}
  {{/evidence}}

  ### Diagnosis Steps

  1. **Review Agent Prompt**
     - Open the agent's system prompt
     - Check for instructions related to: {{category}}

  2. **Analyze Conversation**
     - [View trace]({{trace_url}})
     - Look at agent responses before failure

  3. **Check Tool Usage**
     {{#if category == "agent_no_tool_calls"}}
     Agent didn't use any tools. Add explicit instructions like:
     "When user asks about X, use the get_X_details tool"
     {{/if}}
     {{#if category == "incorrect_tool_calls"}}
     Agent called wrong tools or with wrong args. Add examples:
     "Example: get_reservation_details(reservation_id='ABC123')"
     {{/if}}
     {{#if category == "agent_didnt_communicate"}}
     Agent had the info but didn't tell user. Add:
     "Always explicitly state the [specific info] to the user"
     {{/if}}

  ### Fix Template

  Add to agent system prompt:
  ```
  [INSERT SPECIFIC INSTRUCTION BASED ON CATEGORY]
  ```
```

#### Case: Fix Environment

**Trigger**: `classification.target == "environment"`

```yaml
title: "Environment Fix Required: {{category}} in task {{task_id}}"
priority: P2
team: platform
labels:
  - environment-fix
  - {{domain}}

runbook: |
  ## Environment Issue

  **Category**: {{category}}
  **Evidence**:
  {{#evidence}}
  - {{.}}
  {{/evidence}}

  ### Diagnosis Steps

  1. **Check Tool Execution Logs**
     ```
     service:tau2-bench-agent event:tau2.tool.result task_id:{{task_id}}
     ```

  2. **Verify Environment State**
     - Check initial_state in task definition
     - Verify database fixtures are correct

  3. **Tool Schema Validation**
     {{#if category == "tool_execution_errors"}}
     - Are tool arguments being validated correctly?
     - Is the error message helpful for debugging?
     {{/if}}
     {{#if category == "missing_data"}}
     - Is the required data in the environment?
     - Check task.initial_state.initialization_data
     {{/if}}

  ### Fix Locations

  - Tool definitions: `src/tau2/domains/{{domain}}/tools.py`
  - Environment: `src/tau2/domains/{{domain}}/environment.py`
  - Task fixtures: `data/tasks/{{domain}}/`
```

#### Case: Fix User Simulator

**Trigger**: `classification.target == "user_simulator"`

```yaml
title: "User Simulator Fix: {{category}} in task {{task_id}}"
priority: P3
team: ai-platform
labels:
  - user-sim-fix
  - {{domain}}

runbook: |
  ## User Simulator Issue

  **Category**: {{category}}
  **Evidence**:
  {{#evidence}}
  - {{.}}
  {{/evidence}}

  ### Diagnosis Steps

  1. **Check User LLM Spans**
     - [View LLM Observability]({{llm_obs_url}})
     - Look at user simulator prompts and completions

  2. **Review User Persona**
     - Check task.user_persona
     - Is the persona providing necessary info?

  3. **Analyze User Behavior**
     {{#if category == "user_not_providing_info"}}
     - User may not be giving agent the info needed
     - Check if persona includes required context
     - Verify user_llm prompt includes task goal
     {{/if}}

  ### Fix Template

  Update user persona or simulator prompt:
  ```
  The user should provide the following when asked:
  - [required info 1]
  - [required info 2]
  ```
```

#### Case: Fix Task Definition

**Trigger**: `classification.target == "task"`

```yaml
title: "Task Definition Review: {{task_id}} in {{domain}}"
priority: P4
team: ai-platform
labels:
  - task-review
  - {{domain}}

runbook: |
  ## Task Definition Issue

  **Evidence**:
  {{#evidence}}
  - {{.}}
  {{/evidence}}

  ### Diagnosis Steps

  1. **Compare to Similar Tasks**
     - Find tasks with same tools/goal that pass
     - What's different about this task's assertions?

  2. **Review Assertions**
     - Are NL assertions achievable given task setup?
     - Is db_check expecting correct state?
     - Are action_checks in correct order?

  3. **Check Task Consistency**
     - Does initial_state match expected flow?
     - Are required tools available?

  ### Fix Locations

  - Task definition: `data/tasks/{{domain}}/task_{{task_id}}.json`
  - Assertions: Check `reward_basis` and assertion definitions
```

### Dashboard: Failure Classification Overview

```json
{
  "title": "Failure Classification",
  "widgets": [
    {
      "title": "Fix Target Distribution",
      "type": "sunburst",
      "query": "sum:tau2.failure.classified{*} by {target,category}.as_count()"
    },
    {
      "title": "Agent Issues by Category",
      "type": "toplist",
      "query": "sum:tau2.failure.classified{target:agent} by {category}.as_count()"
    },
    {
      "title": "Recent Agent Fixes Needed",
      "type": "log_stream",
      "query": "service:tau2-bench-agent classification.target:agent",
      "columns": ["task_id", "category", "remediation"]
    }
  ]
}
```

## Summary: From Task Failure to Remediation

```
Task Failure Detected
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  1. DETECTION                                                  │
│     • Metric: tau2.task.reward < 0.7                          │
│     • Log: event:tau2.task.failed                             │
│     • Monitor fires, creates Case                             │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  2. CONTEXT INJECTION                                          │
│     Case includes:                                             │
│     • task_id, domain, evaluation_id                          │
│     • reward_breakdown (which assertions failed)              │
│     • termination_reason                                       │
│     • Link to APM trace (trace_id correlation)                │
│     • Link to stored evaluation JSON                          │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  3. ROOT CAUSE ANALYSIS                                        │
│     Dashboard shows:                                           │
│     • Which assertion type failed (db/action/nl/communicate)  │
│     • Tool call patterns (too many? wrong ones?)              │
│     • Conversation flow in trace                              │
│     • Similar failures in other tasks                         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  4. REMEDIATION                                                │
│     Runbook provides:                                          │
│     • Specific steps based on failure type                    │
│     • Comparison to successful tasks                          │
│     • Prompt review checklist                                  │
│     • Tool schema verification steps                          │
└───────────────────────────────────────────────────────────────┘
```
