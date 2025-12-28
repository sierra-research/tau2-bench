# Datadog Integration Enhancements

## Overview

This document outlines proposed enhancements to the tau2-bench Datadog integration, based on analysis of the current implementation and validation against real simulation data.

**Purpose**: Extend observability from basic task success metrics to comprehensive efficiency and quality analysis.

**Core Narrative**: "Observability for Agentic AI Quality" - monitoring whether the agent accomplished the task correctly, not just whether it responded.

---

## Important: Black-Box Evaluation Context

tau2-bench evaluates agents as **black boxes** via the A2A protocol. This has critical implications for what metrics we can reliably measure:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Evaluation Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────┐           ┌──────────────────────────────┐    │
│  │   User Simulator     │   A2A     │   Agent Under Test           │    │
│  │   (we control)       │◄─────────►│   (black box)                │    │
│  │                      │   HTTP    │                              │    │
│  │  ✅ Actual tokens    │           │  ❌ Unknown model            │    │
│  │  ✅ Actual cost      │           │  ❌ Unknown tokens           │    │
│  │  ✅ Reasoning tokens │           │  ❌ Unknown cost             │    │
│  └──────────────────────┘           └──────────────────────────────┘    │
│                                                                          │
│  What we CAN observe from the agent:                                     │
│  ✅ Response TEXT content (can estimate tokens)                         │
│  ✅ Response LATENCY                                                    │
│  ✅ Number of TURNS in conversation                                     │
│  ✅ Number and correctness of TOOL CALLS                                │
│  ✅ Task OUTCOME (reward, assertions)                                   │
│  ✅ Task DURATION                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### What We Measure vs What We Cannot

| Category | Metric | Source | Reliability |
|----------|--------|--------|-------------|
| **User Simulator** | Token usage | `messages[].usage` | ✅ Actual (litellm) |
| **User Simulator** | Cost | `messages[].cost` | ✅ Actual (if model registered) |
| **User Simulator** | Reasoning tokens | `raw_data.message.provider_specific_fields.reasoning_content` | ✅ Actual (model-dependent) |
| **Agent** | Token usage | N/A | ❌ Not available |
| **Agent** | Cost | N/A | ❌ Not available |
| **Agent** | Response size | `len(content)` | ⚠️ Estimated (chars/4) |
| **Task** | Duration | `simulation.duration` | ✅ Actual |
| **Task** | Conversation turns | `len(messages)` | ✅ Actual |
| **Task** | Tool call accuracy | `reward_info.action_checks` | ✅ Actual |
| **Task** | Outcome quality | `reward_info.reward` | ✅ Actual |

### Narrative Shift: "Task Completion Efficiency"

Instead of claiming to measure "Agent Efficiency" (which requires knowing the agent's internal costs), we measure **"Task Completion Efficiency"** - how efficiently was the task solved from an observable perspective:

- **Conversation length**: Fewer turns = more efficient
- **Time to resolution**: Faster = more efficient
- **Tool call accuracy**: Fewer errors = more efficient
- **Response brevity**: Shorter accurate responses = more efficient

---

## Part 1: Metrics Validation

### 1.1 Data Source Analysis

Validated against **141 real simulations** across 5 domains (airline, mock, retail, telecom, vacation_rental).

#### Available Data Fields

| Field | JSON Path | Status | Notes |
|-------|-----------|--------|-------|
| Prompt tokens | `messages[].usage.prompt_tokens` | ✅ Available | User simulator only |
| Completion tokens | `messages[].usage.completion_tokens` | ✅ Available | User simulator only |
| Cost | `messages[].cost` | ⚠️ $0 for self-hosted | Requires model registration |
| Reasoning content | `messages[].raw_data...reasoning_content` | ✅ Available | User simulator only |
| Reward | `reward_info.reward` | ✅ Available | Task outcome |
| Duration | `simulation.duration` | ✅ Available | End-to-end time |
| Steps | `len(messages)` | ✅ Available | Conversation length |
| Tool calls | `messages[].tool_calls` | ✅ Available | Both user and agent |
| Agent response text | `messages[role=assistant].content` | ✅ Available | Can estimate tokens |

#### Key Finding: Token Data Structure

Token usage is attached to **user role messages** (the simulated user LLM), not assistant messages.
Assistant (agent) messages have `usage: None` because the agent is a black box:

```json
{
  "role": "user",
  "content": "...",
  "usage": {
    "completion_tokens": 719,
    "prompt_tokens": 4267
  },
  "cost": 0.0
}
```

```json
{
  "role": "assistant",
  "content": "I'll help you with that...",
  "usage": null,
  "cost": null
}
```

### 1.2 Efficiency Formula Validation

Tested multiple formulas on 103 successful simulations:

| Formula | Expression | Range | Mean | Verdict |
|---------|------------|-------|------|---------|
| **Simple** | `reward / (tokens / 1000)` | 0.016 - 0.731 | **0.22** | ✅ Recommended |
| Log-scaled | `reward / log(1 + tokens/1000)` | 0.24 - 1.16 | 0.55 | Alternative |
| Reward/Step | `reward / steps` | 0.03 - 0.17 | 0.09 | Supplementary |

**Recommendation**: Use the **simple formula** - it's intuitive (reward per 1K tokens) and has good dynamic range.

### 1.3 Domain Efficiency Comparison

| Domain | Simulations | Success Rate | Avg Tokens | Avg Efficiency |
|--------|-------------|--------------|------------|----------------|
| Mock | 33 | 58% | 3,915 | **0.56** (highest) |
| Retail | 5 | 40% | 7,050 | 0.17 |
| Vacation Rental | 94 | 85% | 18,482 | 0.15 |
| Telecom | 6 | 33% | 41,502 | **0.02** (lowest) |
| Airline | 3 | 0% | 2,960 | N/A |

**Insight**: Complex multi-step tasks (telecom tech support) use 10x more tokens than simple tasks, with lower efficiency.

### 1.4 Reasoning Token Discovery

The Qwen thinking model includes reasoning content that equals ~100% of completion tokens:

```
Completion tokens: 4,388
Est. reasoning tokens: ~4,647 (chars / 4)
Reasoning as % of completion: 106%
```

**Key Insight**: Most tokens are "thinking" not "output" - this is a unique observability opportunity for reasoning models.

### 1.5 Cost Metric Limitation

All simulations show `cost: $0.0` because they use self-hosted Nebius API.

**Options for Demo**:
- **Option A**: Use token count as a proxy for cost
- **Option B**: Apply standard pricing formula: `$0.001/1K input + $0.002/1K output`
- **Option C**: Only demo cost metrics when using paid APIs (OpenAI/Anthropic)

---

## Part 2: New Use Cases

### Use Case 1: Task Completion Efficiency Metrics

**Problem**: Current metrics track quality (reward) separately from how the task was completed, but don't measure overall task efficiency.

**Solution**: Combined efficiency metrics that capture both correctness AND observable resource usage (conversation length, time, tool accuracy).

#### Metric Categories

**Category A: Reliable Metrics (Actual Data)**

| Metric | Formula | Type | Description |
|--------|---------|------|-------------|
| `tau2.task.reward_per_turn` | `reward / conversation_turns` | gauge | Quality per interaction |
| `tau2.task.reward_per_second` | `reward / duration_seconds` | gauge | Quality per time spent |
| `tau2.task.turns_total` | `count(user + assistant msgs)` | gauge | Conversation length |
| `tau2.task.tool_calls_total` | `count(tool_calls)` | gauge | Total tool invocations |
| `tau2.task.tool_accuracy` | `correct_tools / total_tools` | gauge | Tool call correctness |
| `tau2.task.first_attempt_success` | `1 if reward >= 0.7 and turns <= 4` | gauge | Solved quickly |

**Category B: User Simulator Metrics (Actual - Test Harness Cost)**

| Metric | Formula | Type | Description |
|--------|---------|------|-------------|
| `tau2.simulator.tokens_total` | `sum(prompt + completion)` | gauge | User simulator token usage |
| `tau2.simulator.tokens_prompt` | `sum(prompt_tokens)` | gauge | Input tokens to simulator |
| `tau2.simulator.tokens_completion` | `sum(completion_tokens)` | gauge | Output tokens from simulator |
| `tau2.simulator.cost_usd` | `sum(messages[].cost)` | gauge | Simulator cost (if registered) |
| `tau2.simulator.reasoning_tokens` | `len(reasoning_content) / 4` | gauge | Thinking tokens (Qwen/Claude) |

**Category C: Agent Response Metrics (Estimated)**

| Metric | Formula | Type | Description |
|--------|---------|------|-------------|
| `tau2.agent.response_chars` | `sum(len(assistant.content))` | gauge | Agent response verbosity |
| `tau2.agent.est_tokens` | `response_chars / 4` | gauge | Estimated agent tokens |
| `tau2.agent.avg_response_length` | `response_chars / agent_turns` | gauge | Avg response size |

> **Note**: Agent token estimates use the `chars/4` heuristic. These are clearly labeled as estimates and should not be used for cost calculations.

#### Thresholds

See **Appendix A** for validated threshold values based on analysis of 141 real simulations.

#### New Dashboard Widgets

1. **Task Completion Efficiency** - Timeseries of `tau2.task.reward_per_turn` with threshold marker
2. **Conversation Length Distribution** - Histogram of `tau2.task.turns_total` by success/failure
3. **Tool Accuracy by Domain** - Bar chart of `tau2.task.tool_accuracy` grouped by domain
4. **Simulator Token Breakdown** - Stacked bar of prompt vs completion tokens (test harness cost)

#### New Monitor: DR-006

```json
{
  "id": "DR-006",
  "name": "tau2-bench: Low Task Efficiency Alert",
  "type": "metric alert",
  "query": "avg(last_15m):avg:tau2.task.reward_per_turn{env:production} < 0.05",
  "message": "{{#is_alert}}\n**ALERT: Low Task Efficiency Detected**\n\nTasks are requiring too many conversation turns relative to quality achieved.\n\nCurrent Reward/Turn: {{value}}\nThreshold: 0.05\n\nPossible Causes:\n- Agent requiring excessive clarification\n- Inefficient tool call patterns\n- Agent not understanding user intent\n\nRecommended Actions:\n1. Check conversation length distribution\n2. Review tool call accuracy metrics\n3. Examine failed tasks for common patterns\n\n@slack-ai-alerts\n{{/is_alert}}",
  "tags": ["team:ai-platform", "service:tau2-bench", "env:production"],
  "options": {
    "thresholds": {
      "critical": 0.03,
      "warning": 0.05
    }
  }
}
```

---

### Use Case 2: User Simulator Reasoning Analysis

**Problem**: Thinking/reasoning models (Qwen, Claude, o1) used as user simulators spend significant tokens on internal reasoning. This represents test harness cost that should be tracked.

**Solution**: Extract and track reasoning tokens from the user simulator to understand test infrastructure costs.

> **Scope**: This applies to the **user simulator** (which we control), not the agent under test (which is a black box).

#### New Metrics

| Metric | Formula | Type | Description |
|--------|---------|------|-------------|
| `tau2.simulator.reasoning_tokens` | `len(reasoning_content) / 4` | gauge | Simulator thinking tokens |
| `tau2.simulator.reasoning_ratio` | `reasoning_tokens / completion_tokens` | gauge | Thinking vs output ratio |
| `tau2.simulator.output_tokens` | `completion - reasoning` | gauge | Actual simulator output |

#### Dashboard Widget

**"Simulator Thinking vs Output Tokens"** - Stacked bar chart:
- Blue: Output tokens (actual simulator response)
- Orange: Reasoning tokens (simulator thinking)

**Interpretation**: High reasoning ratio indicates the simulator is doing extensive "thinking" before responding. This is expected for thinking models but impacts test harness cost.

#### Implementation Note

Reasoning content is only available for models that expose it via litellm:
- **JSON Path**: `raw_data.message.provider_specific_fields.reasoning_content`
- Qwen thinking models (via Nebius): ✅ Confirmed working
- Claude extended thinking: ✅ Should work (same field structure)
- Standard models (GPT-4, Claude 3.5 non-thinking): ❌ This metric will be 0

---

### Use Case 3: Conversation Flow Analysis

**Problem**: Two tasks with the same reward may have very different conversation quality - one efficient, one with excessive back-and-forth.

**Solution**: Track conversation flow patterns to identify inefficient interactions.

#### Metrics

Uses metrics from Use Case 1:
- `tau2.task.turns_total` - Total conversation turns
- `tau2.task.tool_calls_total` - Total tool invocations
- `tau2.task.tool_accuracy` - Tool call correctness

#### Dashboard Widget

**"Conversation Efficiency"** - Heatmap showing:
- X-axis: Steps taken
- Y-axis: Reward achieved
- Color: Efficiency score (reward/turn)

Helps identify the "sweet spot" of optimal conversation length.

---

### Use Case 4: Domain-Specific Insights

**Problem**: Different domains have different complexity profiles, but current dashboard doesn't highlight this.

**Solution**: Domain-specific efficiency tracking and comparison.

#### New Metrics

| Metric | Formula | Type | Tags |
|--------|---------|------|------|
| `tau2.domain.avg_efficiency` | `avg(efficiency) by domain` | gauge | domain |
| `tau2.domain.cost_per_success` | `sum(tokens) / count(success)` | gauge | domain |
| `tau2.domain.avg_steps` | `avg(steps) by domain` | gauge | domain |

#### Dashboard Widget

**"Domain Comparison"** - Multi-series bar chart:
- Group by domain
- Bars: Success rate, Avg tokens, Avg efficiency

**Story**: "Airline domain requires 2x tokens but has same success rate as retail → opportunity for airline-specific prompt optimization"

---

### Use Case 5: Failure Root Cause Classification

**Problem**: When tasks fail, the current dashboard shows that they failed, but not why.

**Solution**: Correlate assertion failures to identify root cause categories.

#### Root Cause Categories

Based on assertion combination patterns:

| Pattern | Root Cause | Action |
|---------|------------|--------|
| `action_match=false` | Tool selection error | Review agent's tool understanding |
| `action_match=true, db_match=false` | Argument extraction error | Check entity extraction |
| `nl_assertions failed` | Communication failure | Review response generation |
| `max_errors termination` | Schema mismatch | Verify tool definitions |
| `max_steps termination` | Conversation loop | Check user simulator behavior |

#### Dashboard Widget

**"Failure Root Cause Sunburst"** - Nested breakdown:
```
Failed Tasks
├── Tool Selection Errors (action_match=false)
├── Argument Errors (action_match=true, db_match=false)
├── Communication Errors (nl_assertions failed)
├── Max Errors (invalid tool calls)
└── Max Steps (conversation loops)
```

#### New Metrics

| Metric | Type | Tags |
|--------|------|------|
| `tau2.failure.tool_selection` | count | domain |
| `tau2.failure.argument_error` | count | domain |
| `tau2.failure.communication` | count | domain |
| `tau2.failure.max_errors` | count | domain |
| `tau2.failure.max_steps` | count | domain |

---

### Use Case 6: Model Comparison (A/B Testing)

**Problem**: When comparing different LLM backends, there's no unified view of performance differences.

**Solution**: Add model tag to all metrics and create comparison dashboard.

#### Implementation

Add `model` tag extracted from simulation info:
```python
model_tag = f"model:{data['info']['user_info']['llm'].split('/')[-1]}"
```

#### New Dashboard Widgets

**"Model Performance Comparison"** - Side-by-side comparison:
- Success rate by model
- Avg tokens by model
- Avg efficiency by model
- Avg duration by model

**Story**: "Gemini 2.0 Flash has 15% lower reward but uses 60% fewer tokens - which is better for production?"

---

## Part 3: Dashboard Gap Analysis

Before implementing new features, the current dashboard has gaps that need addressing:

### Metrics Referenced But Not Emitted (Broken Widgets)

| Metric | Used In | Status | Resolution |
|--------|---------|--------|------------|
| `tau2.llm.token_cost` | DR-003 Monitor | ❌ Not emitted | Add in Phase 2 (simulator cost) |
| `tau2.llm.tokens_input` | Dashboard LLM Token Usage | ❌ Not emitted | Add in Phase 2 |
| `tau2.llm.tokens_output` | Dashboard LLM Token Usage | ❌ Not emitted | Add in Phase 2 |
| `tau2.task.error_count` | Dashboard Task Error Rate | ❌ Not emitted | Derive from termination counts |
| `tau2.task.failed` | Dashboard Recent Failures | ❌ Not emitted | Log stream, not metric |

### Metrics Emitted But Not In Dashboard (Underutilized)

| Metric | Status | Recommendation |
|--------|--------|----------------|
| `tau2.assertion.nl_failed` | ✅ Emitted | Add to failure analysis widget |
| `tau2.evaluation.avg_reward` | ✅ Emitted | Add evaluation-level widget |
| `tau2.evaluation.pass_rate` | ✅ Emitted | Add evaluation-level widget |
| `tau2.evaluation.tasks_total` | ✅ Emitted | Add to summary |
| `tau2.task.steps` | ✅ Emitted | Already have (rename consideration) |
| `tau2.tool.arguments_match` | ✅ Emitted | Add to tool accuracy widget |
| `tau2.tool.correct` | ✅ Emitted | Add to tool accuracy widget |

### Metric Naming Alignment

The document proposes `tau2.task.turns_total` but code already emits `tau2.task.steps`.

**Decision**: Keep `tau2.task.steps` as-is, add new efficiency metrics alongside.

---

## Part 4: Implementation Plan

### Phase 1: Task Completion Efficiency Metrics

**Priority**: High
**Effort**: Low (1-2 hours)

1. **Update `emit_metrics.py`**:
   - Calculate conversation turns from messages
   - Calculate reward per turn
   - Calculate tool call accuracy from action_checks
   - Emit task efficiency metrics via DogStatsD

2. **Files to modify**:
   - `src/experiments/datadog/scripts/emit_metrics.py`

3. **New metrics emitted** (Category A - Reliable):
   - `tau2.task.reward_per_turn`
   - `tau2.task.reward_per_second`
   - `tau2.task.turns_total`
   - `tau2.task.tool_calls_total`
   - `tau2.task.tool_accuracy`
   - `tau2.task.first_attempt_success`

### Phase 2: User Simulator Metrics + Cost Tracking

**Priority**: High
**Effort**: Medium (2-3 hours)

#### 2a. LiteLLM Model Registration (Enable Cost Tracking)

Currently all costs are $0 because self-hosted models aren't registered with litellm.

**Environment Variable** (already exists in `.env`):
```bash
TAU2_LLM_MODELS={"nebius/Qwen/Qwen3-30B-A3B-Thinking-2507":{"max_tokens":32768,"input_cost_per_token":0.0000001,"output_cost_per_token":0.0000003,"litellm_provider":"nebius"}}
```

**Implementation** - Add to `src/tau2/utils/llm_utils.py` at module load time:
```python
import os
import json

# Register custom models for cost tracking
TAU2_LLM_MODELS = os.getenv("TAU2_LLM_MODELS")
if TAU2_LLM_MODELS:
    try:
        custom_models = json.loads(TAU2_LLM_MODELS)
        litellm.register_model(custom_models)
        logger.info(f"Registered {len(custom_models)} custom model(s) for cost tracking")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse TAU2_LLM_MODELS: {e}")
    except Exception as e:
        logger.warning(f"Failed to register custom models: {e}")
```

**Files to modify**:
- `src/tau2/utils/llm_utils.py` - Add model registration at module load
- `src/tau2/config.py` - Optional: Add config constant for TAU2_LLM_MODELS

#### 2b. Emit Simulator Token Metrics

**Update `emit_metrics.py`**:
- Extract token usage from user messages (role="user")
- Sum prompt and completion tokens
- Emit simulator cost metrics
- Also emit `tau2.llm.tokens_input/output` to fix broken dashboard widgets

**New metrics emitted** (Category B - Test Harness):
- `tau2.simulator.tokens_total`
- `tau2.simulator.tokens_prompt`
- `tau2.simulator.tokens_completion`
- `tau2.simulator.cost_usd`
- `tau2.llm.tokens_input` (fixes dashboard)
- `tau2.llm.tokens_output` (fixes dashboard)
- `tau2.llm.token_cost` (fixes DR-003 monitor)

### Phase 3: Agent Response Estimation (Optional)

**Priority**: Medium
**Effort**: Low (30 min)

1. **Update `emit_metrics.py`**:
   - Sum character length of assistant messages
   - Estimate tokens using chars/4

2. **New metrics emitted** (Category C - Estimated):
   - `tau2.agent.response_chars`
   - `tau2.agent.est_tokens`

3. **Note**: Clearly label these as estimates in dashboard widgets.

### Phase 4: Dashboard Enhancements

**Priority**: High
**Effort**: Low (1 hour)

1. **Update `dashboards.json`**:
   - Add Task Completion Efficiency widget
   - Add Conversation Length Distribution widget
   - Add Tool Accuracy by Domain widget
   - Add Simulator Token Breakdown widget

2. **Files to modify**:
   - `src/experiments/datadog/configs/dashboards.json`

### Phase 5: New Monitor (DR-006)

**Priority**: Medium
**Effort**: Low (30 min)

1. **Update `monitors.json`**:
   - Add DR-006 Low Task Efficiency Alert

2. **Files to modify**:
   - `src/experiments/datadog/configs/monitors.json`

### Phase 6: Simulator Reasoning Analysis

**Priority**: Medium (valuable for thinking model demos)
**Effort**: Medium (1-2 hours)

This tracks the **user simulator's** thinking overhead, not the agent (which is a black box).

1. **Update `emit_metrics.py`**:
   - Extract reasoning content from user messages at:
     `messages[role=user].raw_data.message.provider_specific_fields.reasoning_content`
   - Estimate reasoning tokens via `len(content) / 4`
   - Emit reasoning metrics

2. **New metrics emitted**:
   - `tau2.simulator.reasoning_tokens`
   - `tau2.simulator.reasoning_ratio` (reasoning / completion)

3. **Model Support**:
   - ✅ Qwen thinking models (via Nebius): Confirmed working
   - ✅ Claude extended thinking: Should work (same field)
   - ❌ Standard models: Will emit 0 (no reasoning content)

4. **Implementation snippet**:
```python
def extract_reasoning_tokens(messages: list) -> int:
    """Extract reasoning tokens from user simulator messages."""
    total_reasoning_chars = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        raw_data = msg.get("raw_data", {})
        message = raw_data.get("message", {})
        provider_fields = message.get("provider_specific_fields", {})
        reasoning = provider_fields.get("reasoning_content", "")
        if reasoning:
            total_reasoning_chars += len(reasoning)
    return total_reasoning_chars // 4  # chars/4 estimate
```

### Phase 7: Failure Classification (Optional)

**Priority**: Low
**Effort**: Medium (2 hours)

1. **Update `emit_metrics.py`**:
   - Analyze assertion patterns
   - Classify failure root causes
   - Emit failure category metrics

2. **Update `dashboards.json`**:
   - Add Failure Root Cause Sunburst widget

---

## Part 5: Testing Plan

### Unit Tests

1. **Token extraction**: Verify correct aggregation from messages
2. **Efficiency calculation**: Verify formula with known values
3. **Dry-run mode**: Verify metrics are logged correctly

### Integration Tests

1. Run `emit_metrics.py --dry-run` on existing evaluation files
2. Verify all new metrics are emitted with correct tags
3. Compare calculated efficiency against expected values

### Validation Commands

```bash
# Test token extraction on real data
python emit_metrics.py --evaluation-id eval-1766621026316-44904b --dry-run

# Verify efficiency calculation
python -c "
from emit_metrics import calculate_efficiency
assert calculate_efficiency(1.0, 10000) == 0.1
assert calculate_efficiency(0.5, 5000) == 0.1
print('Efficiency formula validated')
"
```

---

## Part 6: Demo Script Enhancement

### Updated Demo Flow

**Act 1: Normal Operations (Baseline)**
```bash
traffic_generator.py --count 5 --mode normal
```
- Dashboard shows healthy metrics
- **NEW**: Task efficiency widget shows reward/turn ~0.08
- **NEW**: Conversation length histogram shows most tasks < 15 turns
- All monitors green

**Act 2: Quality Regression (Trigger Alert)**
```bash
traffic_generator.py --count 3 --mode failure
```
- Reward drops below 0.7 → DR-002 fires
- **NEW**: Reward per turn drops → DR-006 fires
- **NEW**: Conversation length increases (agent struggling)
- **NEW**: Tool accuracy drops visible in domain chart

**Act 3: Investigation Path**
1. Click into Case → see context
2. **NEW**: Check task efficiency metrics (reward/turn, tool accuracy)
3. **NEW**: Compare conversation length vs successful tasks
4. Drill into LLM Observability → see prompts/completions

### Key Talking Points

1. **"We're not just monitoring if the agent responded — we're monitoring if the agent solved the customer's problem efficiently."**

2. **"Task Completion Efficiency tells you if the agent is both correct AND efficient. A task that succeeds but requires 40 turns instead of 10 indicates the agent is struggling to understand the user."**

3. **"Tool accuracy shows whether the agent is making the right decisions. Low tool accuracy with high reward means the agent is getting lucky; low tool accuracy with low reward means the agent fundamentally misunderstands the task."**

4. **"We clearly separate what we can measure (task outcomes, conversation patterns) from what we can't (agent's internal token usage). This is honest observability for black-box AI systems."**

---

## Appendix A: Validated Thresholds

Based on analysis of 141 real simulations:

### Task Completion Metrics (Reliable)

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| `tau2.task.reward` | >= 0.7 | < 0.7 | < 0.5 |
| `tau2.task.reward_per_turn` | > 0.08 | < 0.05 | < 0.03 |
| `tau2.task.turns_total` | < 15 | > 25 | > 40 |
| `tau2.task.tool_accuracy` | > 0.9 | < 0.7 | < 0.5 |
| `tau2.task.duration_seconds` | < 60 | > 120 | > 300 |

### User Simulator Metrics (Test Harness Cost)

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| `tau2.simulator.tokens_total` | < 20,000 | > 30,000 | > 50,000 |
| `tau2.simulator.reasoning_ratio` | < 1.0 | > 1.5 | > 2.0 |

### Agent Response Metrics (Estimated)

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| `tau2.agent.response_chars` | < 5,000 | > 10,000 | > 20,000 |
| `tau2.agent.est_tokens` | < 1,250 | > 2,500 | > 5,000 |

---

## Appendix B: Metric Reliability Guide

| Category | Metrics | Data Source | Use For |
|----------|---------|-------------|---------|
| **A: Task Outcomes** | reward, turns, duration, tool_accuracy | Direct measurement | Primary efficiency analysis |
| **B: Simulator** | tokens_total, cost_usd, reasoning_ratio | litellm actual usage | Test harness cost tracking |
| **C: Agent Estimates** | response_chars, est_tokens | chars/4 heuristic | Relative comparison only |

**Do NOT**:
- Use Category C metrics for cost calculations
- Claim to know agent's actual token usage
- Compare Category C metrics across different agent types

**DO**:
- Use Category A metrics for efficiency comparisons
- Track Category B to understand test infrastructure costs
- Use Category C for relative verbosity comparisons within same agent

---

## Change Log

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-28 | Claude | Initial draft based on simulation analysis |
| 2025-12-28 | Claude | Updated for black-box evaluation context; renamed metrics to reflect what we can actually measure |
| 2025-12-28 | Claude | Added dashboard gap analysis, fixed reasoning_content path, added litellm model registration |
