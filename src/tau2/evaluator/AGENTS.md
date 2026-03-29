# AGENTS.md — src/tau2/evaluator/

## Overview

The evaluator module scores simulation runs. Each evaluator is a classmethod-only class that takes a task and trajectory, and returns a `RewardInfo`.

## Architecture

### Evaluator Types

| Evaluator | What it checks | `RewardType` |
|-----------|---------------|-------------|
| `EnvironmentEvaluator` | DB state matches gold after replaying actions | `DB`, `ENV_ASSERTION` |
| `ActionEvaluator` | Agent called the right tools with right args | `ACTION` |
| `CommunicateEvaluator` | Agent said required information to user | `COMMUNICATE` |
| `NLAssertionsEvaluator` | LLM-judged natural language assertions (WIP) | `NL_ASSERTION` |

### Half-Duplex vs Full-Duplex Variants

Every evaluator has TWO variants:

- `FooEvaluator(EvaluatorBase[Message])` — works on `list[Message]` (half-duplex)
- `FullDuplexFooEvaluator(EvaluatorBase[Tick])` — works on `list[Tick]` (full-duplex)

Full-duplex evaluators convert `Tick` lists to `Message` lists internally before evaluating. Each has a `ticks_to_message_history()` classmethod that handles this conversion. The conversion logic differs per evaluator:
- **Environment**: extracts tool calls in execution order (user first, then agent per tick)
- **Communicate**: merges agent chunks by overlapping `utterance_ids`
- **NL Assertions**: uses containment-aware linearization from the streaming module
- **Action**: extracts all tool calls (user + agent) from ticks

### Composition via `evaluate_simulation()`

The `evaluate_simulation()` function in `evaluator.py` is the main entry point. It:
1. Selects half-duplex vs full-duplex evaluator variants based on `CommunicationMode`
2. Runs the appropriate evaluators based on `EvaluationType`
3. Combines rewards multiplicatively based on the task's `reward_basis`

### `EvaluationType` Controls What Gets Evaluated

- `ALL` — runs ENV + ACTION + COMMUNICATE, but only includes each in the final reward if it's in the task's `reward_basis`
- `ALL_IGNORE_BASIS` — runs all three and always multiplies them together
- `ENV`, `ACTION`, `COMMUNICATE` — run a single evaluator
- `NL_ASSERTIONS`, `ALL_WITH_NL_ASSERTIONS` — include NL assertions (WIP)
- `ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS` — like `ALL_IGNORE_BASIS` but also includes NL assertions (WIP)

## Rules When Modifying This Module

1. **Always maintain both variants**: If you change an evaluator, update BOTH the half-duplex and full-duplex versions.

2. **`calculate_reward` is a classmethod**: All evaluators use `@classmethod`. No instance state.

3. **Reward is multiplicative**: The combined reward is the product of component rewards. A 0 in any component zeroes the total.

4. **Handle missing criteria gracefully**: If `task.evaluation_criteria` is `None` or a specific criterion is absent, return `reward=1.0` (not 0). Missing criteria means "not evaluated", not "failed".

5. **Premature termination = 0 reward**: If `simulation.termination_reason` is not `AGENT_STOP` or `USER_STOP`, the simulation gets reward 0 before any evaluators run.

6. **Environment evaluator needs `environment_constructor` and `env_kwargs`**: Unlike other evaluators, `EnvironmentEvaluator.calculate_reward()` takes an `environment_constructor` callable to create fresh environments for gold vs predicted comparison. It also accepts `env_kwargs: dict` for domain-specific parameters (e.g., `retrieval_variant` for `banking_knowledge`). These kwargs are threaded from `evaluate_simulation()` through to both the predicted and gold environment constructors. Always pass `env_kwargs` when calling `calculate_reward()` to ensure domains with custom constructor parameters work correctly.

7. **NL assertions require LLM calls**: `NLAssertionsEvaluator` calls an LLM (configured via `DEFAULT_LLM_NL_ASSERTIONS` in `config.py`). This is experimental/WIP.

8. **Hallucination check**: `check_hallucination()` and `format_hallucination_feedback()` in `reviewer.py` are used by the runner's hallucination retry loop (full-duplex only). The underlying LLM judge is in `hallucination_reviewer.py`. See `tau2.runner.batch` and the `--hallucination-retries` CLI option.

Consider these rules if they affect your changes.
